"""Sparse IDK metrics. Only requested pairwise results are dense, never the basis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pyidk import pairwise_similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

from cartpole_idk.storage.embeddings import EmbeddingSet

DEFAULT_MAX_PAIRS = 4_000_000


def check_pairwise_size(n: int, m: int, max_pairs: int = DEFAULT_MAX_PAIRS) -> None:
    """Reject dense pairwise allocations larger than the configured limit.

    Args:
        n: Number of query rows.
        m: Number of reference rows.
        max_pairs: Maximum allowed matrix entries.
    """
    if max_pairs < 1 or n * m > max_pairs:
        raise ValueError(
            f"Pairwise result needs {n * m:,} entries (limit {max_pairs:,}); "
            "reduce the population/increase stride or explicitly raise max_pairs"
        )


def _inputs(
    x: csr_matrix, y: csr_matrix | None, t: int, max_pairs: int
) -> tuple[csr_matrix, csr_matrix]:
    """Normalize sparse inputs and validate dimensions and pairwise allocation size.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
    """
    x = csr_matrix(x, dtype=float)
    y = x if y is None else csr_matrix(y, dtype=float)
    if t < 1 or x.shape[1] != y.shape[1] or x.shape[1] % t:
        raise ValueError("Require positive t and matching embedding widths divisible by t")
    if not np.isfinite(x.data).all() or not np.isfinite(y.data).all():
        raise ValueError("Embeddings must be finite")
    check_pairwise_size(x.shape[0], y.shape[0], max_pairs)
    return x, y


def pairwise_idk_similarity(
    x: csr_matrix, y: csr_matrix | None = None, *, t: int, max_pairs: int = DEFAULT_MAX_PAIRS
) -> np.ndarray:
    """Native pyidk dot-product/t similarity; self-similarity need not equal one.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
    """
    x, y = _inputs(x, y, t, max_pairs)
    return pairwise_similarity(x, y, n_partitions=t, dense=True)


def pairwise_idk_distance(
    x: csr_matrix, y: csr_matrix | None = None, *, t: int, max_pairs: int = DEFAULT_MAX_PAIRS
) -> np.ndarray:
    """Kernel-induced distance = ||mu_x - mu_y|| / sqrt(t), roundoff clipped.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
    """
    self_pair = y is None or y is x
    x, y = _inputs(x, y, t, max_pairs)
    xx = np.asarray(x.multiply(x).sum(axis=1)).ravel() / t
    yy = np.asarray(y.multiply(y).sum(axis=1)).ravel() / t
    squared = xx[:, None] + yy[None, :] - 2 * pairwise_similarity(x, y, n_partitions=t)
    result = np.sqrt(np.maximum(squared, 0))
    if self_pair:
        np.fill_diagonal(result, 0)
    return result


def pairwise_cosine(
    x: csr_matrix, y: csr_matrix | None = None, *, t: int = 1, max_pairs: int = DEFAULT_MAX_PAIRS
) -> np.ndarray:
    """Cosine similarity. Every comparison involving a zero vector is zero.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Partition count for shape validation; cosine needs no t normalization.
        max_pairs: Maximum entries in the dense query-by-reference result.
    """
    x, y = _inputs(x, y, t, max_pairs)
    norms = (
        np.sqrt(np.asarray(x.multiply(x).sum(axis=1)))
        * np.sqrt(np.asarray(y.multiply(y).sum(axis=1))).T
    )
    dots = (x @ y.T).toarray()
    return np.clip(np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0), -1, 1)


def _occupancy(x: csr_matrix, partition: int, psi: int) -> np.ndarray:
    # Densify one partition at a time, not all t * psi features.
    """Recover one partition's probabilities, including its outside-region mass.

    Args:
        x: Sparse mean-occupancy embedding rows.
        partition: Zero-based isolation partition index.
        psi: Number of regions per partition.
    """
    block = x[:, partition * psi : (partition + 1) * psi].toarray()
    mass = block.sum(axis=1)
    if np.any(block < -1e-10) or np.any(mass > 1 + 1e-10):
        raise ValueError("Divergences require nonnegative mean occupancy with partition mass <= 1")
    block = np.maximum(block, 0)
    augmented = np.column_stack((block, np.maximum(0, 1 - block.sum(axis=1))))
    # Only correct floating-point drift, never renormalize away outside mass.
    return augmented / augmented.sum(axis=1, keepdims=True)


def _divergence(
    x: csr_matrix, y: csr_matrix | None, *, t: int, epsilon: float | None, max_pairs: int
) -> np.ndarray:
    """Compute partition-averaged JS or smoothed KL divergence in bounded blocks.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
        epsilon: KL smoothing mass per cell; None selects unsmoothed JS.
    """
    x, y = _inputs(x, y, t, max_pairs)
    psi = x.shape[1] // t
    if psi < 1:
        raise ValueError("Occupancy requires at least one cell per partition")
    result = np.zeros((x.shape[0], y.shape[0]))
    for partition in range(t):
        p, q = _occupancy(x, partition, psi), _occupancy(y, partition, psi)
        if epsilon is not None:
            p = (p + epsilon) / (1 + epsilon * (psi + 1))
            q = (q + epsilon) / (1 + epsilon * (psi + 1))
        # Bound broadcasting memory to approximately 64 * 64 * (psi+1) values.
        for i in range(0, x.shape[0], 64):
            for j in range(0, y.shape[0], 64):
                a, b = p[i : i + 64, None, :], q[None, j : j + 64, :]
                if epsilon is None:
                    # SciPy returns sqrt(JS); this API returns JS divergence in nats.
                    values = jensenshannon(a, b, axis=-1) ** 2
                else:
                    values = rel_entr(a, b).sum(axis=-1)
                result[i : i + 64, j : j + 64] += values / t
    return np.maximum(result, 0)


def pairwise_js_divergence(
    x: csr_matrix, y: csr_matrix | None = None, *, t: int, max_pairs: int = DEFAULT_MAX_PAIRS
) -> np.ndarray:
    """Mean partition JS divergence (natural logs), including outside occupancy.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
    """
    return _divergence(x, y, t=t, epsilon=None, max_pairs=max_pairs)


def pairwise_kl_divergence(
    x: csr_matrix,
    y: csr_matrix | None = None,
    *,
    t: int,
    epsilon: float,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> np.ndarray:
    """KL(query || reference), averaged over partitions, with explicit smoothing.

    Add epsilon to every cell INCLUDING outside, then renormalize. Asymmetric.

    Args:
        x: Query embedding rows, with t * psi columns.
        y: Reference embedding rows; None compares x with itself.
        t: Number of isolation partitions.
        max_pairs: Maximum entries in the dense query-by-reference result.
        epsilon: Positive mass added to every occupancy cell before renormalization.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("KL requires finite epsilon > 0")
    return _divergence(x, y, t=t, epsilon=epsilon, max_pairs=max_pairs)


@dataclass(frozen=True, slots=True)
class Metric:
    """Pairwise metric implementation, preferred ranking direction, and display description."""

    function: Callable[..., np.ndarray]
    direction: Literal["higher", "lower"]
    description: str


METRICS = {
    "idk": Metric(pairwise_idk_similarity, "higher", "Native IDK dot product / t"),
    "idk-distance": Metric(pairwise_idk_distance, "lower", "IDK-induced distance"),
    "cosine": Metric(pairwise_cosine, "higher", "Cosine; zero-norm comparisons are zero"),
    "js": Metric(pairwise_js_divergence, "lower", "Mean partition JS divergence in nats"),
    "kl": Metric(pairwise_kl_divergence, "lower", "Smoothed KL(query || reference) in nats"),
}


def pairwise(
    query: EmbeddingSet,
    reference: EmbeddingSet | None = None,
    *,
    metric: str = "idk",
    max_pairs: int = DEFAULT_MAX_PAIRS,
    **parameters: Any,
) -> np.ndarray:
    """Dispatch using metric metadata and enforce shared fitting-space identity.

    Args:
        query: Query embeddings defining result rows.
        reference: Compatible reference embeddings defining columns; None uses query.
        metric: Registered similarity or distance name.
        max_pairs: Maximum entries in the dense result.
        **parameters: Metric-specific arguments, such as positive epsilon for KL.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if reference is not None:
        query.compatible_with(reference)
    if metric == "kl" and "epsilon" not in parameters:
        raise ValueError("KL requires explicit epsilon smoothing")
    return METRICS[metric].function(
        query.values,
        None if reference is None else reference.values,
        t=query.t,
        max_pairs=max_pairs,
        **parameters,
    )
