"""Small external-validity diagnostics, applied only after unsupervised clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score


def cluster_purity(labels: np.ndarray, categories: np.ndarray) -> float | None:
    """Fraction in the dominant category of each cluster; missing categories excluded."""
    frame = pd.DataFrame({"cluster": labels, "category": categories}).dropna()
    if frame.empty:
        return None
    counts = frame.groupby(["cluster", "category"]).size()
    return float(counts.groupby(level=0).max().sum() / len(frame))


def evaluate_clusters(
    assignments: pd.DataFrame,
    distances: np.ndarray | None = None,
    *,
    categorical: tuple[str, ...] = ("perturbation_type", "checkpoint_id"),
    exclude_noise: bool = True,
) -> dict[str, Any]:
    """Exclude all negative noise labels by default. Undefined silhouette is null."""
    labels = assignments["cluster"].to_numpy()
    keep = labels >= 0 if exclude_noise else np.ones(len(labels), dtype=bool)
    selected = labels[keep]
    report: dict[str, Any] = {
        "noise_excluded": exclude_noise,
        "n_evaluated": int(keep.sum()),
        "n_noise": int((labels < 0).sum()),
        "silhouette": None,
    }
    if distances is not None:
        if distances.shape != (len(labels), len(labels)):
            raise ValueError("Distance matrix shape must match assignments")
        if 1 < len(np.unique(selected)) < len(selected):
            report["silhouette"] = float(
                silhouette_score(distances[np.ix_(keep, keep)], selected, metric="precomputed")
            )
    for column in categorical:
        if column not in assignments:
            continue
        valid = keep & assignments[column].notna().to_numpy()
        categories = assignments.loc[valid, column].astype(str).to_numpy()
        report[column] = {
            "adjusted_mutual_information": float(
                adjusted_mutual_info_score(labels[valid], categories)
            )
            if valid.any()
            else None,
            "purity": cluster_purity(labels[valid], categories),
        }
    return report


def cluster_summary(
    assignments: pd.DataFrame,
    *,
    categorical: tuple[str, ...] = ("perturbation_type", "checkpoint_id"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-cluster return statistics and tidy category counts, including noise."""
    summaries, composition = [], []
    for cluster, frame in assignments.groupby("cluster", sort=True):
        row: dict[str, Any] = {"cluster": cluster, "count": len(frame)}
        if "episode_return" in frame:
            values = frame.episode_return.dropna().to_numpy(dtype=float)
            row.update(
                return_count=len(values),
                mean_return=float(values.mean()) if len(values) else None,
                median_return=float(np.median(values)) if len(values) else None,
                std_return=float(values.std()) if len(values) else None,
            )
        summaries.append(row)
        for column in categorical:
            if column in frame:
                for value, count in (
                    frame[column].fillna("<missing>").astype(str).value_counts().items()
                ):
                    composition.append(
                        {
                            "cluster": cluster,
                            "field": column,
                            "value": value,
                            "count": count,
                            "fraction": count / len(frame),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(
        composition, columns=["cluster", "field", "value", "count", "fraction"]
    )
