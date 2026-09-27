from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cartpole_idk.model import InitialStateConfig


@dataclass(slots=True)
class TrainingConfig:
    """DQN hyperparameters and training schedule; step counts refer to environment interactions."""

    total_steps: int = 200_000
    max_episode_steps: int = 500
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    eval_every: int = 5_000
    eval_episodes: int = 20
    checkpoint_every: int = 10_000
    seed: int = 42
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return settings with JSON-compatible initial-state parameters."""
        settings = asdict(self)
        settings["initial_state"] = self.initial_state.model_dump(mode="json")
        return settings

    def save(self, path: Path) -> None:
        """Write training settings to an indented JSON file.

        Args:
            path: Destination configuration file.
        """
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
