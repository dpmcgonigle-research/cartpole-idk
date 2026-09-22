"""Sparse, already-created embeddings and their provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from cartpole_idk.storage.units import AnalysisUnit


@dataclass(slots=True)
class EmbeddingSet:
    """Sparse IDK vectors and their ordered analysis-unit provenance.

    Each row corresponds to one unit; t partitions with psi regions each give
    t * psi columns. basis_id identifies the shared fitted scaler and basis.
    """

    units: list[AnalysisUnit]
    values: csr_matrix
    basis_id: str
    t: int
    psi: int
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize to CSR and validate dimensions, finite values, and unique unit IDs."""
        self.values = csr_matrix(self.values, dtype=float)
        if self.t < 1 or self.psi < 2 or self.values.shape != (len(self.units), self.t * self.psi):
            raise ValueError("Embedding shape must match units and t * psi")
        if not np.isfinite(self.values.data).all():
            raise ValueError("Embeddings must be finite")
        if len({u.unit_id for u in self.units}) != len(self.units):
            raise ValueError("Unit IDs must be unique")

    def subset(self, indices: list[int]) -> EmbeddingSet:
        """Select aligned embedding rows and units in the requested order.

        Args:
            indices: Zero-based row indices to retain.
        """
        return EmbeddingSet(
            [self.units[i] for i in indices], self.values[indices], self.basis_id, self.t, self.psi
        )

    def compatible_with(self, other: EmbeddingSet) -> None:
        """Reject populations that use different fitted spaces or embedding dimensions.

        Args:
            other: Embedding population to compare against this one.
        """
        if (self.basis_id, self.t, self.psi) != (other.basis_id, other.t, other.psi):
            raise ValueError("Both populations must use the same fitted basis and scaler")
