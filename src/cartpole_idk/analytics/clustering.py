"""Scikit-learn clustering adapters; labels never enter representation fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import BayesianGaussianMixture

from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, pairwise
from cartpole_idk.model import ClusterConfig
from cartpole_idk.storage.embeddings import EmbeddingSet


@dataclass(slots=True)
class ClusterResult:
    """Unit assignments and labels, plus algorithm-specific distances or diagnostics.

    Negative labels mark noise; probabilities are HDBSCAN membership strengths
    or mixture responsibilities, depending on the selected method.
    """

    assignments: pd.DataFrame
    labels: np.ndarray
    distances: np.ndarray | None = None
    coordinates: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    components: dict[str, Any] = field(default_factory=dict)


def cluster_units(
    embedded: EmbeddingSet,
    config: ClusterConfig | None = None,
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    precomputed: np.ndarray | None = None,
) -> ClusterResult:
    """HDBSCAN/DBSCAN use distances; spectral uses native IDK affinity.

    JS clustering uses sqrt(mean partition JS), not divergence itself. DPGMM
    uses sparse TruncatedSVD followed by sklearn's DP-style Bayesian mixture.
    A supplied precomputed matrix must follow the selected algorithm's semantics.

    Args:
        embedded: Unit embeddings and metadata in matching row order.
        config: Algorithm settings; omitted uses default HDBSCAN.
        max_pairs: Maximum entries allowed in a computed dense pairwise matrix.
        precomputed: Optional N-by-N distances for density methods or IDK affinity
            for spectral clustering; JS distances must already be square-rooted.
    """
    config = config or ClusterConfig()
    n = len(embedded.units)
    if n < 2:
        raise ValueError("Clustering requires at least two units")
    table = pd.DataFrame([u.record() for u in embedded.units])
    distances, coordinates, probabilities = None, None, None
    components: dict[str, Any] = {}
    matrix = precomputed
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Precomputed matrix must be finite, nonnegative and N by N")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Precomputed matrix must be symmetric")
    if config.algorithm in {"hdbscan", "dbscan"}:
        if config.distance not in {"idk-distance", "js"}:
            raise ValueError("Clustering distance must be idk-distance or js")
        distances = (
            pairwise(embedded, metric=config.distance, max_pairs=max_pairs)
            if matrix is None
            else matrix.copy()
        )
        if matrix is None and config.distance == "js":
            np.sqrt(distances, out=distances)
        if not np.allclose(np.diag(distances), 0):
            raise ValueError("A distance matrix must have a zero diagonal")
        np.fill_diagonal(distances, 0)
        if config.algorithm == "hdbscan":
            model = HDBSCAN(
                metric="precomputed",
                min_cluster_size=config.min_cluster_size,
                min_samples=config.min_samples,
                cluster_selection_method=config.cluster_selection_method,
                copy=True,
            )
            labels = model.fit_predict(distances)
            probabilities = model.probabilities_
            table["membership_strength"] = probabilities
        else:
            labels = DBSCAN(
                metric="precomputed",
                eps=config.eps,
                min_samples=5 if config.min_samples is None else config.min_samples,
            ).fit_predict(distances)
    elif config.algorithm == "spectral":
        affinity = (
            pairwise(embedded, metric="idk", max_pairs=max_pairs) if matrix is None else matrix
        )
        labels = SpectralClustering(
            n_clusters=config.n_clusters, affinity="precomputed", random_state=config.random_state
        ).fit_predict(affinity)
    elif config.algorithm == "dpgmm":
        if matrix is not None:
            raise ValueError("DPGMM operates on embeddings, not a precomputed matrix")
        if not 1 <= config.svd_components <= min(embedded.values.shape):
            raise ValueError("svd_components must be between 1 and min(n_units, embedding_width)")
        if not 1 <= config.n_components <= n:
            raise ValueError("n_components must be between 1 and n_units")
        reducer = TruncatedSVD(n_components=config.svd_components, random_state=config.random_state)
        coordinates = reducer.fit_transform(embedded.values)
        mixture = BayesianGaussianMixture(
            n_components=config.n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=config.weight_concentration_prior,
            random_state=config.random_state,
            max_iter=config.max_iter,
        ).fit(coordinates)
        probabilities = mixture.predict_proba(coordinates)
        labels = probabilities.argmax(axis=1)
        table["component_probability"] = probabilities.max(axis=1)
        components = {
            "weights": mixture.weights_,
            "means": mixture.means_,
            "covariances": mixture.covariances_,
            "converged": bool(mixture.converged_),
            "iterations": int(mixture.n_iter_),
            "svd_components": reducer.components_,
            "svd_explained_variance_ratio": reducer.explained_variance_ratio_,
        }
    else:
        raise ValueError(f"Unknown clustering algorithm: {config.algorithm}")
    table["cluster"] = labels
    table["is_noise"] = labels < 0
    return ClusterResult(table, labels, distances, coordinates, probabilities, components)
