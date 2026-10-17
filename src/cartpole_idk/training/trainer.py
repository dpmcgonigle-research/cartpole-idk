from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cartpole_idk.initial_state import reset_cartpole
from cartpole_idk.training.checkpoint import save_checkpoint
from cartpole_idk.training.config import TrainingConfig
from cartpole_idk.training.dqn import DQNAgent, ReplayBuffer, Transition
from cartpole_idk.training.evaluation import EvaluationFailure, evaluate_agent

logger = logging.getLogger(__name__)


def _epsilon(step: int, cfg: TrainingConfig) -> float:
    """Linearly anneal exploration to its configured final value.

    Args:
        step: Current training environment-step count.
        cfg: Exploration endpoints and decay duration.
    """
    fraction = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def _append_csv(path: Path, row: dict) -> None:
    """Append one metric record, writing a header when creating the file.

    Args:
        path: Destination CSV file.
        row: Ordered metric names and values for one observation.
    """
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _log_evaluation(
    run_dir: Path,
    cfg: TrainingConfig,
    *,
    step: int,
    kind: str,
    seed: int,
    metrics: dict[str, float],
    failures: list[EvaluationFailure] | None,
) -> None:
    """Log aggregate metrics and optionally append a section of failed initial states."""
    logger.info(
        "Step %d/%d: %s evaluation: mean return=%.2f, std=%.2f, "
        "fraction reaching %d steps=%.4f, p10=%.2f, p25=%.2f, p50=%.2f, p75=%.2f, p90=%.2f",
        step,
        cfg.total_steps,
        kind,
        metrics["mean_return"],
        metrics["std_return"],
        cfg.max_episode_steps,
        metrics["full_length_rate"],
        metrics["p10_return"],
        metrics["p25_return"],
        metrics["p50_return"],
        metrics["p75_return"],
        metrics["p90_return"],
    )
    if failures is None:
        return
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"\n=== {kind} evaluation: step={step}, time={timestamp} ===",
        f"episodes={cfg.eval_episodes} seed={seed} max_episode_steps={cfg.max_episode_steps} "
        f"failures={len(failures)}",
    ]
    for failure in failures:
        lines.append(
            f"episode={failure.episode} seed={failure.seed} return={failure.episode_return:.17g} "
            f"x={failure.x:.17g} x_dot={failure.x_dot:.17g} "
            f"theta={failure.theta:.17g} theta_dot={failure.theta_dot:.17g}"
        )
    if not failures:
        lines.append("No failures (all returns reached max_episode_steps).")
    with (run_dir / "failure_init_conds.log").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def train(run_dir: str | Path, cfg: TrainingConfig, *, device: str = "cpu") -> Path:
    """Train a CartPole DQN and return the directory holding metrics and checkpoints.

    Args:
        run_dir: Destination directory for the training run.
        cfg: Training, evaluation, initialization, exploration, and checkpoint settings.
        device: PyTorch device used for policy training and inference.
    """
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
    obs = reset_cartpole(env, cfg.initial_state, seed=cfg.seed)
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
            obs = reset_cartpole(env, cfg.initial_state)
            episode_return, episode_length = 0.0, 0

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            logger.info(
                "Step %d/%d: evaluating over %d episodes", step, cfg.total_steps, cfg.eval_episodes
            )
            failures: list[EvaluationFailure] | None = [] if cfg.debug_init_conds else None
            evaluation_seed = cfg.seed + 100_000 + step
            metrics = evaluate_agent(
                agent,
                episodes=cfg.eval_episodes,
                max_episode_steps=cfg.max_episode_steps,
                seed=evaluation_seed,
                initial_state=cfg.initial_state,
                failures=failures,
            )
            _append_csv(run_dir / "evaluation_metrics.csv", {"step": step, **metrics})
            _log_evaluation(
                run_dir,
                cfg,
                step=step,
                kind="periodic",
                seed=evaluation_seed,
                metrics=metrics,
                failures=failures,
            )

        if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
            logger.info("Step %d/%d: evaluating checkpoint", step, cfg.total_steps)
            failures = [] if cfg.debug_init_conds else None
            evaluation_seed = cfg.seed + 200_000 + step
            metrics = evaluate_agent(
                agent,
                episodes=cfg.eval_episodes,
                max_episode_steps=cfg.max_episode_steps,
                seed=evaluation_seed,
                initial_state=cfg.initial_state,
                failures=failures,
            )
            _log_evaluation(
                run_dir,
                cfg,
                step=step,
                kind="checkpoint",
                seed=evaluation_seed,
                metrics=metrics,
                failures=failures,
            )
            save_checkpoint(
                agent,
                run_dir / "checkpoints" / f"step_{step:09d}.pt",
                step=step,
                metadata={"training_config": cfg.to_dict(), "evaluation": metrics},
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
