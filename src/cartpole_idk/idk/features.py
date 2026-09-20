from __future__ import annotations

import numpy as np
from pyidk import SequenceBatch
from pyidk.representation import TransitionRepresentation, WindowRepresentation

from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.storage import Trajectory


def _observations(traj: Trajectory, source: str) -> np.ndarray:
    return traj.true_observations if source == "true" else traj.agent_observations


def build_sequence_batch(
    trajectories: list[Trajectory], config: IDKExperimentConfig
) -> SequenceBatch:
    sequences: list[np.ndarray] = []
    for traj in trajectories:
        obs = _observations(traj, config.observation_source)
        actions = (
            traj.commanded_actions if config.action_source == "commanded" else traj.executed_actions
        )

        if config.representation in {"state", "transition", "window"}:
            sequences.append(obs)
        elif config.representation == "state_action":
            one_hot = np.eye(2, dtype=float)[actions]
            sequences.append(np.concatenate([obs[:-1], one_hot], axis=1))
        elif config.representation == "state_action_next_state":
            one_hot = np.eye(2, dtype=float)[actions]
            sequences.append(np.concatenate([obs[:-1], one_hot, obs[1:]], axis=1))
        else:
            raise ValueError(f"Unknown representation: {config.representation}")

    batch = SequenceBatch.from_sequences(sequences)
    if config.representation == "transition":
        batch = TransitionRepresentation(lags=(0, 1)).transform(batch)
    elif config.representation == "window":
        batch = WindowRepresentation(length=config.window_length).transform(batch)
    return batch
