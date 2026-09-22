from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from cartpole_idk.generation.perturbations import Perturbation
from cartpole_idk.model import GeneratedTrajectory, GenerationReport
from cartpole_idk.storage import Trajectory, TrajectoryStore
from cartpole_idk.training import load_checkpoint

logger = logging.getLogger(__name__)


def _trajectory_id() -> str:
    """Create a timestamped, randomized identifier for a generated rollout."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"traj_{stamp}_{secrets.token_hex(4)}"


def generate_dataset(
    *,
    checkpoint,
    output,
    episodes: int,
    seed: int,
    max_episode_steps: int = 500,
    perturbation: Perturbation | None = None,
    device: str = "cpu",
) -> TrajectoryStore:
    """Run a greedy checkpoint policy and save trajectories plus a generation report.

    Args:
        checkpoint: Policy checkpoint to load.
        output: Destination trajectory dataset directory.
        episodes: Number of rollouts to record.
        seed: Base episode seed, incremented for successive rollouts.
        max_episode_steps: Maximum transitions per rollout.
        perturbation: Action/observation transformation; None uses nominal behavior.
        device: PyTorch device for policy inference.
    """
    agent, payload = load_checkpoint(checkpoint, device=device)
    checkpoint_path = Path(checkpoint)
    store = TrajectoryStore(output)
    store.initialize()
    perturbation = perturbation or Perturbation()
    rng = np.random.default_rng(seed)
    generated: list[GeneratedTrajectory] = []
    progress_every = max(1, episodes // 10)
    logger.info(
        "Starting generation: %d episodes, perturbation=%s, seed=%d, output=%s",
        episodes,
        perturbation.name,
        seed,
        output,
    )

    for episode in range(episodes):
        env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
        true_obs, _ = env.reset(seed=seed + episode)
        pmeta = perturbation.reset(rng, max_episode_steps)
        true_observations = [np.asarray(true_obs, dtype=np.float32)]
        agent_observations, commanded_actions, executed_actions = [], [], []
        rewards, terminated_flags, truncated_flags = [], [], []
        step = 0
        while True:
            agent_obs = perturbation.agent_observation(true_obs, step)
            commanded = agent.act(agent_obs, epsilon=0.0)
            executed = perturbation.executed_action(commanded, step)
            next_true, reward, terminated, truncated, _ = env.step(executed)
            agent_observations.append(np.asarray(agent_obs, dtype=np.float32))
            commanded_actions.append(commanded)
            executed_actions.append(executed)
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            true_observations.append(np.asarray(next_true, dtype=np.float32))
            true_obs = next_true
            step += 1
            if terminated or truncated:
                break

        agent_observations.append(perturbation.agent_observation(true_obs, step))
        tid = _trajectory_id()
        metadata = {
            "checkpoint_id": checkpoint_path.stem,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": int(payload.get("step", -1)),
            "episode_index": episode,
            "environment_seed": seed + episode,
            "generation_seed": seed,
            "perturbation_type": perturbation.name,
            "perturbation_onset": pmeta.get("onset"),
            "perturbation": pmeta,
            "max_episode_steps": max_episode_steps,
        }
        traj = Trajectory(
            tid,
            np.stack(true_observations),
            np.stack(agent_observations),
            np.asarray(commanded_actions, dtype=np.int8),
            np.asarray(executed_actions, dtype=np.int8),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated_flags, dtype=bool),
            np.asarray(truncated_flags, dtype=bool),
            metadata,
        )
        store.add(traj)
        generated.append(
            GeneratedTrajectory(
                trajectory_id=tid,
                episode_return=traj.episode_return,
                length=traj.length,
                perturbation_onset=pmeta.get("onset"),
            )
        )
        env.close()
        if (episode + 1) % progress_every == 0 or episode + 1 == episodes:
            logger.info(
                "Generation progress: %d/%d episodes saved; last return=%.1f, length=%d",
                episode + 1,
                episodes,
                traj.episode_return,
                traj.length,
            )

    GenerationReport(
        checkpoint=str(checkpoint_path),
        episodes=episodes,
        seed=seed,
        perturbation_type=perturbation.name,
        generated=generated,
    ).to_file(Path(output, "generation_report.json"))
    logger.info(
        "Generation complete: %d trajectories saved; report=%s",
        len(generated),
        Path(output, "generation_report.json"),
    )
    return store
