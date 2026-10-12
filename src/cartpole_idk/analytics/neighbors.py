"""Neighbor retrieval and rolling nominal/failure likeness as tidy tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyidk import reduce_references

from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, METRICS, PYIDK_METRICS
from cartpole_idk.storage.embeddings import EmbeddingSet
from cartpole_idk.storage.units import AnalysisUnit


def eligible_references(
    query: Sequence[AnalysisUnit],
    reference: Sequence[AnalysisUnit],
    *,
    exclude_same_unit: bool = True,
    exclude_same_trajectory: bool = False,
    max_overlap: float | None = None,
) -> np.ndarray:
    """Build one eligibility block using original source identities and intervals."""
    allowed = np.ones((len(query), len(reference)), dtype=bool)
    if not exclude_same_unit and not exclude_same_trajectory and max_overlap is None:
        return allowed
    same_source = np.ones_like(allowed)
    for part in (0, 1):
        left = np.asarray([unit.source_key[part] for unit in query], dtype=object)
        right = np.asarray([unit.source_key[part] for unit in reference], dtype=object)
        same_source &= left[:, None] == right[None, :]
    a = np.asarray([u.metadata.get("source_start_step", u.start_step) for u in query])[:, None]
    b = np.asarray([u.metadata.get("source_end_step", u.end_step) for u in query])[:, None]
    c = np.asarray([u.metadata.get("source_start_step", u.start_step) for u in reference])[None, :]
    d = np.asarray([u.metadata.get("source_end_step", u.end_step) for u in reference])[None, :]
    if exclude_same_unit:
        left_ids = np.asarray([u.unit_id for u in query], dtype=object)
        right_ids = np.asarray([u.unit_id for u in reference], dtype=object)
        allowed &= ~(left_ids[:, None] == right_ids[None, :])
        allowed &= ~(same_source & (a == c) & (b == d))
    if exclude_same_trajectory:
        allowed &= ~same_source
    if max_overlap is not None:
        left_lengths = np.asarray([u.raw_length for u in query])[:, None]
        right_lengths = np.asarray([u.raw_length for u in reference])[None, :]
        overlap = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
        allowed &= ~(
            same_source & (overlap > max_overlap * np.minimum(left_lengths, right_lengths))
        )
    return allowed


@dataclass(slots=True)
class NeighborResult:
    """Ranked query/reference matches and one aggregate score row per query unit."""

    neighbors: pd.DataFrame
    scores: pd.DataFrame


def top_k_neighbors(
    query: EmbeddingSet,
    reference: EmbeddingSet,
    *,
    metric: str = "idk",
    k: int = 5,
    exclude_same_unit: bool = True,
    exclude_same_trajectory: bool = False,
    max_overlap: float | None = None,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    metric_parameters: dict[str, Any] | None = None,
) -> NeighborResult:
    """Stable rank by metric direction; cap k at eligible references per query.

    max_overlap excludes same-source intervals whose intersection divided by the
    shorter interval exceeds this threshold. No eligible neighbors yields count=0
    and a missing mean score, rather than an invented familiarity value.

    Args:
        query: Units to score, with their embeddings.
        reference: Candidate neighbors embedded with the same fitted model.
        metric: Registered metric controlling both comparison and ranking direction.
        k: Maximum number of eligible neighbors per query.
        exclude_same_unit: Exclude identical units or matching source intervals.
        exclude_same_trajectory: Exclude every unit from the same original source trajectory.
        max_overlap: Maximum intersection/shorter-length ratio for same-source intervals.
        max_pairs: Maximum entries in a working pairwise block (not the total population).
        metric_parameters: Extra metric arguments, such as epsilon for KL.
    """
    if k < 1 or (max_overlap is not None and not 0 <= max_overlap <= 1):
        raise ValueError("Require k >= 1 and max_overlap in [0, 1]")
    query.compatible_with(reference)
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    parameters = dict(metric_parameters or {})
    if metric in {"js", "kl"}:
        parameters.setdefault("atol", 1e-10)

    def eligibility(q_indices: np.ndarray, r_indices: np.ndarray) -> np.ndarray:
        return eligible_references(
            [query.units[i] for i in q_indices],
            [reference.units[i] for i in r_indices],
            exclude_same_unit=exclude_same_unit,
            exclude_same_trajectory=exclude_same_trajectory,
            max_overlap=max_overlap,
        )

    query_block = min(128, max_pairs)
    result = reduce_references(
        query.values,
        reference.values,
        n_partitions=query.t,
        metric=PYIDK_METRICS[metric],
        metric_parameters=parameters,
        aggregation="top_k",
        k=k,
        eligibility=eligibility,
        query_block_size=query_block,
        reference_block_size=min(1024, max_pairs // query_block),
        return_selected=True,
    )
    assert result.selected_indices is not None and result.selected_values is not None
    links, scores = [], []
    for i, unit in enumerate(query.units):
        selected = result.selected_indices[i]
        scores.append(
            {
                **unit.record(),
                "metric": metric,
                "neighbor_count": len(selected),
                "score": float(result.scores[i]) if len(selected) else None,
            }
        )
        for rank, (j, value) in enumerate(zip(selected, result.selected_values[i], strict=True), 1):
            links.append(
                {
                    "query_unit_id": unit.unit_id,
                    "rank": rank,
                    "metric": metric,
                    "metric_value": float(value),
                    **{
                        f"reference_{key}": value
                        for key, value in reference.units[j].record().items()
                    },
                }
            )
    return NeighborResult(
        pd.DataFrame(
            links,
            columns=None
            if links
            else [
                "query_unit_id",
                "rank",
                "metric",
                "metric_value",
                "reference_unit_id",
                "reference_trajectory_id",
            ],
        ),
        pd.DataFrame(scores),
    )


def reference_likeness(
    query: EmbeddingSet,
    nominal: EmbeddingSet,
    failure: EmbeddingSet,
    *,
    metric: str = "idk",
    **options: Any,
) -> tuple[pd.DataFrame, dict[str, NeighborResult]]:
    """Keep both scores; positive margin always means more nominal-like.

    Args:
        query: Units to compare against both populations.
        nominal: Nominal reference embeddings.
        failure: Failure reference embeddings.
        metric: Registered metric used for both comparisons.
        **options: Neighbor selection and metric settings passed to top_k_neighbors.
    """
    results = {
        "nominal": top_k_neighbors(query, nominal, metric=metric, **options),
        "failure": top_k_neighbors(query, failure, metric=metric, **options),
    }
    table = results["nominal"].scores.drop(columns=["score", "neighbor_count"]).copy()
    for name, result in results.items():
        table[f"{name}_score"] = result.scores["score"].astype(float)
        table[f"{name}_count"] = result.scores["neighbor_count"]
    sign = 1 if METRICS[metric].direction == "higher" else -1
    table["margin"] = sign * (table.nominal_score - table.failure_score)
    return table, results


def rolling_likeness(
    query: EmbeddingSet,
    references: dict[str, EmbeddingSet],
    *,
    metrics: tuple[str, ...] = ("idk",),
    **options: Any,
) -> pd.DataFrame:
    """One row per query window, metric and reference group; no plotting or refitting.

    Args:
        query: Precomputed window embeddings to score in their stored order.
        references: Named reference populations sharing the query's fitted model.
        metrics: Registered metrics to compute for each reference group.
        **options: Neighbor selection and metric settings passed to top_k_neighbors.
    """
    tables = []
    for metric in metrics:
        for group, reference in references.items():
            table = top_k_neighbors(query, reference, metric=metric, **options).scores
            table["reference_group"] = group
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
