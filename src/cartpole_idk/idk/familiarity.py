from __future__ import annotations

import numpy as np
from pyidk import reduce_references

from cartpole_idk.idk.features import build_sequence_batch
from cartpole_idk.idk.fitting import IDKReference
from cartpole_idk.storage import Trajectory


def trajectory_familiarity(
    reference: IDKReference, trajectories: list[Trajectory], *, k: int = 5
) -> np.ndarray:
    """Return mean top-k reference similarity for each query trajectory.

    Args:
        reference: Fitted scaler, basis, and reference embeddings to reuse.
        trajectories: Raw query trajectories to transform and score.
        k: Maximum reference neighbors averaged per query.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    batch = build_sequence_batch(trajectories, reference.config)
    scaled = batch.with_values(reference.scaler.transform(batch.values))
    embeddings = reference.model.transform(scaled)
    return reduce_references(
        embeddings,
        reference.reference_embeddings,
        n_partitions=reference.config.t,
        metric="similarity",
        aggregation="top_k",
        k=k,
    ).scores
