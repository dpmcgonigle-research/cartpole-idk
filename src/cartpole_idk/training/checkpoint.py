from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from cartpole_idk.training.dqn import DQNAgent

logger = logging.getLogger(__name__)


def save_checkpoint(agent: DQNAgent, path: Path, *, step: int, metadata: dict[str, Any]) -> None:
    """Save policy weights, architecture, and provenance; optimizer state is not included.

    Args:
        agent: DQN policy whose online-network weights are saved.
        path: Destination checkpoint file; parent directories are created.
        step: Training environment-step count at checkpoint time.
        metadata: Training settings and evaluation information to retain.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "hidden_sizes": list(agent.hidden_sizes),
            "gamma": agent.gamma,
            "model_state": agent.online.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    logger.info("Saved checkpoint at step %d: %s", step, path)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[DQNAgent, dict]:
    """Load policy weights into an agent and return it with the checkpoint payload.

    Args:
        path: Saved policy checkpoint file.
        device: PyTorch device on which to reconstruct the policy.
    """
    logger.info("Loading checkpoint %s on %s", path, device)
    payload = torch.load(path, map_location=device, weights_only=False)
    agent = DQNAgent(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_sizes=tuple(payload["hidden_sizes"]),
        learning_rate=1e-3,
        gamma=float(payload["gamma"]),
        seed=0,
        device=device,
    )
    agent.online.load_state_dict(payload["model_state"])
    agent.target.load_state_dict(payload["model_state"])
    return agent, payload
