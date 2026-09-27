"""Basis-support diagnostics on existing IDK occupancies, without fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cartpole_idk.model import ReturnStatistics, SupportSummary
from cartpole_idk.storage.embeddings import EmbeddingSet

ROUNDING_TOLERANCE = 1e-10
SUPPORT_METRICS = (
    "outside_mass_mean",
    "outside_mass_median",
    "outside_mass_min",
    "outside_mass_max",
    "outside_mass_std",
    "partitions_with_any_outside",
    "support_rate_mean",
)


def partition_outside_mass(embeddings: csr_matrix, *, t: int, psi: int) -> np.ndarray:
    """Return an n_units-by-t array of missing partition occupancy.

    Cells are never renormalized. Only out-of-range roundoff up to 1e-10 is
    clipped; larger violations are rejected. Only partition sums become dense.

    Args:
        embeddings: Sparse mean occupancies with t * psi columns.
        t: Number of partitions from the saved fit configuration.
        psi: Number of isolation regions per partition from that configuration.
    """
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in (t, psi)):
        raise ValueError("t and psi must be positive integers")
    values = csr_matrix(embeddings, dtype=float, copy=True)
    if values.shape[1] != t * psi:
        raise ValueError(f"Embedding width {values.shape[1]} != t * psi ({t * psi})")
    values.sum_duplicates()
    if not np.isfinite(values.data).all():
        raise ValueError("Embedding occupancies must be finite")
    if np.any(values.data < -ROUNDING_TOLERANCE) or np.any(values.data > 1 + ROUNDING_TOLERANCE):
        raise ValueError("Embedding cell occupancy must be in [0, 1]")
    np.clip(values.data, 0, 1, out=values.data)
    columns = np.arange(t * psi)
    grouping = csr_matrix((np.ones(len(columns)), (columns, columns // psi)), shape=(t * psi, t))
    assigned = (values @ grouping).toarray()
    if np.any(assigned > 1 + ROUNDING_TOLERANCE):
        raise ValueError("Assigned partition mass exceeds one; expected mean occupancies")
    return 1 - np.clip(assigned, 0, 1)


def summarize_support(outside_mass: np.ndarray) -> pd.DataFrame:
    """Summarize each unit across partitions, using population standard deviation.

    Args:
        outside_mass: Finite n_units-by-t missing occupancies in [0, 1].
    """
    values = np.asarray(outside_mass, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Outside mass must have shape (n_units, t) with t > 0")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Outside mass must be finite and in [0, 1]")
    means = values.mean(axis=1)
    return pd.DataFrame(
        {
            "outside_mass_mean": means,
            "outside_mass_median": np.median(values, axis=1),
            "outside_mass_min": values.min(axis=1),
            "outside_mass_max": values.max(axis=1),
            "outside_mass_std": values.std(axis=1),
            "partitions_with_any_outside": np.count_nonzero(values > 0, axis=1),
            "support_rate_mean": 1 - means,
        }
    )


def support_dataframe(embedded: EmbeddingSet, outside_mass: np.ndarray) -> pd.DataFrame:
    """Attach support scores to unit provenance without reordering embedding rows.

    Args:
        embedded: Already-loaded embeddings with ordered unit metadata.
        outside_mass: Partition scores aligned with embedded, shape (n_units, t).
    """
    if outside_mass.shape != (len(embedded.units), embedded.t):
        raise ValueError("Outside mass shape must match the embedding units and partitions")
    table = pd.DataFrame([unit.record() for unit in embedded.units])
    table["source_trajectory_id"] = [unit.source_key[1] for unit in embedded.units]
    table["embedding_row"] = np.arange(len(embedded.units))
    for name, values in summarize_support(outside_mass).items():
        table[name] = values.to_numpy()
    return table


def support_summary(table: pd.DataFrame, *, t: int, psi: int) -> SupportSummary:
    """Describe per-unit support metrics with equal weight for every unit.

    Args:
        table: Per-unit table produced by support_dataframe or summarize_support.
        t: Number of fitted partitions.
        psi: Number of fitted regions per partition.
    """
    return SupportSummary(
        n_units=len(table),
        t=t,
        psi=psi,
        statistics={
            name: ReturnStatistics.from_returns(table[name].tolist()) for name in SUPPORT_METRICS
        },
    )


def support_group_summary(table: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Describe each support metric within metadata groups, retaining missing groups.

    Args:
        table: Support scores with the requested provenance column.
        group_by: Scalar metadata column to group on; no labels are assumed.
    """
    if group_by not in table:
        raise ValueError(f"Unknown group-by column: {group_by}")
    if not table[group_by].map(pd.api.types.is_scalar).all():
        raise ValueError(f"Group-by column must contain scalar values: {group_by}")
    rows = []
    for value, group in table.groupby(group_by, sort=False, dropna=False, observed=True):
        for name in SUPPORT_METRICS:
            stats = ReturnStatistics.from_returns(group[name].tolist())
            assert stats is not None
            rows.append(
                {
                    "group_by": group_by,
                    "group_value": value,
                    "n_units": len(group),
                    "metric": name,
                    **stats.model_dump(),
                }
            )
    return pd.DataFrame(rows)
