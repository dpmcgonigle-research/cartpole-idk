from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from cartpole_idk.initial_state import reset_cartpole
from cartpole_idk.model import InitialStateConfig
from cartpole_idk.training.dqn import DQNAgent


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One failed evaluation episode and its exact sampled environment state."""

    episode: int  # One-based index within this evaluation.
    seed: int
    episode_return: float
    x: float
    x_dot: float
    theta: float
    theta_dot: float


def evaluate_agent(
    agent: DQNAgent,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    initial_state: InitialStateConfig | None = None,
    failures: list[EvaluationFailure] | None = None,
) -> dict[str, float]:
    """Summarize greedy-policy returns and lengths without updating network weights.

    Args:
        agent: Policy to evaluate.
        episodes: Number of evaluation rollouts.
        max_episode_steps: Maximum transitions per rollout.
        seed: Base environment seed, incremented for each episode.
        initial_state: Starting-state settings; None uses standard uniform bounds.
        failures: Optional collector for episodes with return below max_episode_steps.
            Records the actual initial environment state before any actions, rather
            than the observation rounded to float32. Existing entries are retained.
    """
    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("episodes and max_episode_steps must be positive")
    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    initial_state = initial_state or InitialStateConfig()
    returns, lengths = [], []
    try:
        for i in range(episodes):
            obs = reset_cartpole(env, initial_state, seed=seed + i)
            start = (
                np.array(cast(CartPoleEnv, env.unwrapped).state, dtype=float, copy=True)
                if failures is not None
                else None
            )
            total, length = 0.0, 0
            while True:
                action = agent.act(obs, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(length)
            if failures is not None and total < max_episode_steps:
                assert start is not None
                x, x_dot, theta, theta_dot = map(float, start)
                failures.append(
                    EvaluationFailure(i + 1, seed + i, total, x, x_dot, theta, theta_dot)
                )
    finally:
        env.close()
    arr, lens = np.asarray(returns), np.asarray(lengths)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "median_return": float(p50),
        "mean_episode_length": float(lens.mean()),
        "full_length_rate": float(np.mean(lens >= max_episode_steps)),
        "p10_return": float(p10),
        "p25_return": float(p25),
        "p50_return": float(p50),
        "p75_return": float(p75),
        "p90_return": float(p90),
    }
