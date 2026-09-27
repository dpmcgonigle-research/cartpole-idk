from __future__ import annotations

import gymnasium as gym
import numpy as np

from cartpole_idk.initial_state import reset_cartpole
from cartpole_idk.model import InitialStateConfig
from cartpole_idk.training.dqn import DQNAgent


def evaluate_agent(
    agent: DQNAgent,
    *,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    initial_state: InitialStateConfig | None = None,
) -> dict[str, float]:
    """Summarize greedy-policy returns and lengths without updating network weights.

    Args:
        agent: Policy to evaluate.
        episodes: Number of evaluation rollouts.
        max_episode_steps: Maximum transitions per rollout.
        seed: Base environment seed, incremented for each episode.
        initial_state: Starting-state settings; None uses standard uniform bounds.
    """
    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    initial_state = initial_state or InitialStateConfig()
    returns, lengths = [], []
    for i in range(episodes):
        obs = reset_cartpole(env, initial_state, seed=seed + i)
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
    env.close()
    arr, lens = np.asarray(returns), np.asarray(lengths)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "median_return": float(np.median(arr)),
        "mean_episode_length": float(lens.mean()),
        "full_length_rate": float(np.mean(lens >= max_episode_steps)),
    }
