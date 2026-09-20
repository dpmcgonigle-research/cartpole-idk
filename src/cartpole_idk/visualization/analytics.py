"""Optional exploratory plots; quantitative analytics do not depend on matplotlib."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def plot_pairwise(matrix: np.ndarray, *, title: str = "Pairwise metric", ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    artist = ax.imshow(matrix, aspect="auto")
    ax.figure.colorbar(artist, ax=ax)
    ax.set(xlabel="Reference unit index", ylabel="Query unit index", title=title)
    return ax


def plot_rolling(
    table: pd.DataFrame, *, scores: tuple[str, ...] = ("score",), ax: Any = None
) -> Any:
    """One curve per trajectory, metric, reference group and score column."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    group_keys = [key for key in ("trajectory_id", "metric", "reference_group") if key in table]
    groups = table.groupby(group_keys, dropna=False) if group_keys else [("query", table)]
    for group, rows in groups:
        rows = rows.sort_values("end_step")
        for score in scores:
            ax.plot(rows.end_step, rows[score], label=f"{group}: {score}")
    ax.set(xlabel="Window end (exclusive)", ylabel="Metric value")
    ax.legend()
    return ax


def plot_cluster_composition(
    composition: pd.DataFrame, *, field: str = "perturbation_type", ax: Any = None
) -> Any:
    counts = (
        composition[composition.field == field]
        .pivot(index="cluster", columns="value", values="count")
        .fillna(0)
    )
    return counts.plot.bar(stacked=True, ax=ax, ylabel="Unit count")


def plot_embedding_2d(
    embeddings: csr_matrix,
    *,
    labels: np.ndarray | None = None,
    random_state: int = 42,
    ax: Any = None,
) -> tuple[Any, np.ndarray]:
    """Sparse SVD display coordinates only; this projection is not clustering evidence."""
    import matplotlib.pyplot as plt

    if min(embeddings.shape) < 2:
        raise ValueError("A 2-D display needs at least two units and two features")
    coordinates = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(embeddings)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels)
    ax.set(xlabel="SVD coordinate 1", ylabel="SVD coordinate 2")
    return ax, coordinates
