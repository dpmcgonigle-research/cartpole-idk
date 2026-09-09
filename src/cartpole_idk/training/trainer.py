from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .dqn import DQNAgent, ReplayBuffer, Transition
from .evaluation import evaluate_agent

logger = logging.getLogger(__name__)


def _epsilon(step: int, cfg: TrainingConfig) -> float:
    fraction = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def _append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(run_dir: str | Path, cfg: TrainingConfig, *, device: str = "cpu") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    logger.info(
        "Starting training: %d steps, seed=%d, device=%s, output=%s",
        cfg.total_steps,
        cfg.seed,
        device,
        run_dir,
    )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = gym.make("CartPole-v1", max_episode_steps=cfg.max_episode_steps)
    obs, _ = env.reset(seed=cfg.seed)
    agent = DQNAgent(4, 2, cfg.hidden_sizes, cfg.learning_rate, cfg.gamma, cfg.seed, device)
    replay = ReplayBuffer(cfg.replay_capacity, cfg.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    last_loss = float("nan")
    progress_every = max(1, cfg.total_steps // 10)

    for step in range(1, cfg.total_steps + 1):
        eps = _epsilon(step, cfg)
        action = agent.act(obs, epsilon=eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay.add(
            Transition(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                done,
            )
        )
        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        if (
            step >= cfg.learning_starts
            and step % cfg.train_frequency == 0
            and len(replay) >= cfg.batch_size
        ):
            last_loss = agent.update(replay.sample(cfg.batch_size))
        if step % cfg.target_update_every == 0:
            agent.sync_target()

        if done:
            _append_csv(
                run_dir / "training_metrics.csv",
                {
                    "step": step,
                    "episode": episode_index,
                    "return": episode_return,
                    "length": episode_length,
                    "epsilon": eps,
                    "loss": last_loss,
                },
            )
            episode_index += 1
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            logger.info(
                "Step %d/%d: evaluating over %d episodes", step, cfg.total_steps, cfg.eval_episodes
            )
            metrics = evaluate_agent(
                agent,
                episodes=cfg.eval_episodes,
                max_episode_steps=cfg.max_episode_steps,
                seed=cfg.seed + 100_000 + step,
            )
            _append_csv(run_dir / "evaluation_metrics.csv", {"step": step, **metrics})
            logger.info(
                "Evaluation: mean return=%.2f, std=%.2f",
                metrics["mean_return"],
                metrics["std_return"],
            )

        if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
            logger.info("Step %d/%d: evaluating checkpoint", step, cfg.total_steps)
            metrics = evaluate_agent(
                agent,
                episodes=cfg.eval_episodes,
                max_episode_steps=cfg.max_episode_steps,
                seed=cfg.seed + 200_000 + step,
            )
            save_checkpoint(
                agent,
                run_dir / "checkpoints" / f"step_{step:09d}.pt",
                step=step,
                metadata={"training_config": asdict(cfg), "evaluation": metrics},
            )
        if step % progress_every == 0 or step == cfg.total_steps:
            logger.info(
                "Training progress: %d/%d steps, %d episodes, epsilon=%.3f, loss=%.4f",
                step,
                cfg.total_steps,
                episode_index,
                eps,
                last_loss,
            )
    env.close()
    logger.info(
        "Training complete: %d steps; metrics and checkpoints in %s", cfg.total_steps, run_dir
    )
    return run_dir
