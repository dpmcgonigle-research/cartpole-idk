from __future__ import annotations

import numpy as np

from cartpole_idk.storage import Trajectory

from .features import build_sequence_batch
from .fitting import IDKReference


def trajectory_familiarity(
    reference: IDKReference, trajectories: list[Trajectory], *, k: int = 5
) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")
    batch = build_sequence_batch(trajectories, reference.config)
    scaled = batch.with_values(reference.scaler.transform(batch.values))
    embeddings = reference.model.transform(scaled)
    similarities = reference.model.similarity(
        embeddings, reference.reference_embeddings, dense=True
    )
    k_eff = min(k, similarities.shape[1])
    top = np.partition(similarities, -k_eff, axis=1)[:, -k_eff:]
    return top.mean(axis=1)
