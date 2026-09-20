from __future__ import annotations

import gymnasium as gym
import numpy as np

from cartpole_idk.training.dqn import DQNAgent


def evaluate_agent(
    agent: DQNAgent, *, episodes: int, max_episode_steps: int, seed: int
) -> dict[str, float]:
    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    returns, lengths = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
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
