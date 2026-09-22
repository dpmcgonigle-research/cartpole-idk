from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Trajectory:
    """Recorded episode or prepared segment, with arrays and source metadata.

    True and agent observations have T+1 rows; actions, rewards, and ending flags
    have T rows. Commanded actions may differ from actions executed after perturbation.
    """

    trajectory_id: str
    true_observations: np.ndarray
    agent_observations: np.ndarray
    commanded_actions: np.ndarray
    executed_actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        """Number of recorded transitions."""
        return int(self.executed_actions.shape[0])

    @property
    def episode_return(self) -> float:
        """Sum of this stored trajectory's rewards, including for prepared segments."""
        return float(self.rewards.sum())

    def save(self, path: Path) -> None:
        """Write compressed arrays and JSON metadata without pickling.

        Args:
            path: Destination NPZ file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            trajectory_id=np.asarray(self.trajectory_id),
            true_observations=self.true_observations,
            agent_observations=self.agent_observations,
            commanded_actions=self.commanded_actions,
            executed_actions=self.executed_actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            metadata_json=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        """Reconstruct trajectory arrays and metadata from an NPZ file.

        Args:
            path: Previously saved trajectory file.
        """
        with np.load(path, allow_pickle=False) as z:
            return cls(
                trajectory_id=str(z["trajectory_id"].item()),
                true_observations=z["true_observations"],
                agent_observations=z["agent_observations"],
                commanded_actions=z["commanded_actions"],
                executed_actions=z["executed_actions"],
                rewards=z["rewards"],
                terminated=z["terminated"],
                truncated=z["truncated"],
                metadata=json.loads(str(z["metadata_json"].item())),
            )
