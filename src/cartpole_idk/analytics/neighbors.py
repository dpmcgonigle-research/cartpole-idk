"""Neighbor retrieval and rolling nominal/failure likeness as tidy tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cartpole_idk.analytics.embeddings import EmbeddingSet
from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, METRICS, pairwise


@dataclass(slots=True)
class NeighborResult:
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
    """
    if k < 1 or (max_overlap is not None and not 0 <= max_overlap <= 1):
        raise ValueError("Require k >= 1 and max_overlap in [0, 1]")
    values = pairwise(
        query, reference, metric=metric, max_pairs=max_pairs, **(metric_parameters or {})
    )
    higher = METRICS[metric].direction == "higher"
    links, scores = [], []
    for i, unit in enumerate(query.units):
        eligible = np.ones(len(reference.units), dtype=bool)
        for j, other in enumerate(reference.units):
            same_source = unit.source_key == other.source_key
            if exclude_same_unit and (
                unit.unit_id == other.unit_id
                or (
                    same_source
                    and unit.metadata.get("source_start_step", unit.start_step)
                    == other.metadata.get("source_start_step", other.start_step)
                    and unit.metadata.get("source_end_step", unit.end_step)
                    == other.metadata.get("source_end_step", other.end_step)
                )
            ):
                eligible[j] = False
            if same_source and exclude_same_trajectory:
                eligible[j] = False
            if same_source and max_overlap is not None:
                start = max(
                    unit.metadata.get("source_start_step", unit.start_step),
                    other.metadata.get("source_start_step", other.start_step),
                )
                end = min(
                    unit.metadata.get("source_end_step", unit.end_step),
                    other.metadata.get("source_end_step", other.end_step),
                )
                if max(0, end - start) / min(unit.raw_length, other.raw_length) > max_overlap:
                    eligible[j] = False
        indices = np.flatnonzero(eligible)
        order = np.argsort(-values[i, indices] if higher else values[i, indices], kind="stable")
        selected = indices[order[:k]]
        scores.append(
            {
                **unit.record(),
                "metric": metric,
                "neighbor_count": len(selected),
                "score": float(values[i, selected].mean()) if len(selected) else None,
            }
        )
        for rank, j in enumerate(selected, 1):
            links.append(
                {
                    "query_unit_id": unit.unit_id,
                    "rank": rank,
                    "metric": metric,
                    "metric_value": float(values[i, j]),
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
    """Keep both scores; positive margin always means more nominal-like."""
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
    """One row per query window, metric and reference group; no plotting or refitting."""
    tables = []
    for metric in metrics:
        for group, reference in references.items():
            table = top_k_neighbors(query, reference, metric=metric, **options).scores
            table["reference_group"] = group
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
