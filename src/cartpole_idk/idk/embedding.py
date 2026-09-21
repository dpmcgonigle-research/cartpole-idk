"""CartPole representation filtering before fit or transform."""

from __future__ import annotations

import numpy as np
from pyidk import SequenceBatch

from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.idk.features import build_sequence_batch
from cartpole_idk.idk.units import UnitCollection


def represented_units(
    collection: UnitCollection,
    config: IDKExperimentConfig,
) -> tuple[UnitCollection, SequenceBatch]:
    if not collection.units:
        raise ValueError("No valid analysis units; check trajectory lengths and selections")
    trajectories = [u.trajectory for u in collection.units]
    if any(t is None for t in trajectories):
        raise ValueError("Raw trajectories are required for representation construction")
    batch = build_sequence_batch([t for t in trajectories if t is not None], config)
    keep = np.flatnonzero(batch.lengths > 0)
    skipped = list(collection.skipped)
    skipped.extend(
        {"unit_id": u.unit_id, "trajectory_id": u.trajectory_id, "reason": "empty_representation"}
        for u, n in zip(collection.units, batch.lengths, strict=True)
        if n == 0
    )
    if not len(keep):
        raise ValueError(
            "All units produce empty representations; reduce representation window length"
        )
    valid = UnitCollection([collection.units[i] for i in keep], skipped)
    if len(keep) != len(collection.units):
        batch = SequenceBatch.from_sequences(batch.sequence(int(i)) for i in keep)
    return valid, batch
