"""Fit once, then transform arbitrary populations without refitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
from pyidk import SequenceBatch
from scipy.sparse import csr_matrix

from cartpole_idk.analytics.units import AnalysisUnit, UnitCollection
from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.idk.features import build_sequence_batch
from cartpole_idk.idk.fitting import IDKReference, fit_sequence_batch


@dataclass(slots=True)
class EmbeddingSet:
    units: list[AnalysisUnit]
    values: csr_matrix
    basis_id: str
    t: int
    psi: int
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.values = csr_matrix(self.values, dtype=float)
        if self.t < 1 or self.psi < 2 or self.values.shape != (len(self.units), self.t * self.psi):
            raise ValueError("Embedding shape must match units and t * psi")
        if not np.isfinite(self.values.data).all():
            raise ValueError("Embeddings must be finite")
        if len({u.unit_id for u in self.units}) != len(self.units):
            raise ValueError("Unit IDs must be unique")

    def subset(self, indices: list[int]) -> EmbeddingSet:
        return EmbeddingSet(
            [self.units[i] for i in indices], self.values[indices], self.basis_id, self.t, self.psi
        )

    def compatible_with(self, other: EmbeddingSet) -> None:
        if (self.basis_id, self.t, self.psi) != (other.basis_id, other.t, other.psi):
            raise ValueError("Both populations must use the same fitted basis and scaler")


def represented_units(
    collection: UnitCollection,
    config: IDKExperimentConfig,
) -> tuple[UnitCollection, SequenceBatch]:
    if not collection.units:
        raise ValueError("No valid analysis units; check trajectory lengths and selections")
    batch = build_sequence_batch([u.trajectory for u in collection.units], config)
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


@dataclass(slots=True)
class AnalysisBasis:
    reference: IDKReference
    fitting: EmbeddingSet

    def transform(self, collection: UnitCollection) -> EmbeddingSet:
        valid, batch = represented_units(collection, self.reference.config)
        scaled = batch.with_values(self.reference.scaler.transform(batch.values))
        return EmbeddingSet(
            valid.units,
            self.reference.model.transform(scaled),
            self.fitting.basis_id,
            self.fitting.t,
            self.fitting.psi,
            valid.skipped,
        )


def fit_embeddings(collection: UnitCollection, config: IDKExperimentConfig) -> AnalysisBasis:
    """Fit on selected units only; no labels or later query values enter fitting."""
    valid, batch = represented_units(collection, config)
    reference = fit_sequence_batch(batch, [u.unit_id for u in valid.units], config)
    fitting = EmbeddingSet(
        valid.units,
        reference.reference_embeddings,
        uuid4().hex,
        config.t,
        config.psi,
        valid.skipped,
    )
    return AnalysisBasis(reference, fitting)
