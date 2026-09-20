"""Population MMD over distribution embeddings, distinct from individual IDK distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import vstack
from sklearn.metrics.pairwise import euclidean_distances

from cartpole_idk.analytics.embeddings import EmbeddingSet
from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, check_pairwise_size


@dataclass(frozen=True, slots=True)
class MMDResult:
    statistic: float
    bandwidth: float
    estimator: str
    n_a: int
    n_b: int
    p_value: float | None
    n_permutations: int
    random_state: int


def population_mmd(
    a: EmbeddingSet,
    b: EmbeddingSet,
    *,
    bandwidth: float | None = None,
    estimator: Literal["biased", "unbiased"] = "biased",
    n_permutations: int = 0,
    random_state: int = 42,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> MMDResult:
    """RBF MMD squared on IDK embedding vectors, k=exp(-||x-y||²/(2 sigma²)).

    Default sigma is the median positive pooled pairwise Euclidean distance;
    use 1 when all vectors coincide. Biased MMD includes diagonals and is
    nonnegative; unbiased MMD excludes them and may be negative. The same pooled
    bandwidth is frozen throughout permutations. Permutation units are embedding
    rows: meaningful p-values require exchangeability (overlapping windows are
    generally dependent). Use whole trajectories or independent units for inference.
    """
    a.compatible_with(b)
    na, nb = len(a.units), len(b.units)
    if not na or not nb or estimator not in {"biased", "unbiased"}:
        raise ValueError("Require nonempty groups and biased/unbiased estimator")
    if estimator == "unbiased" and min(na, nb) < 2:
        raise ValueError("Unbiased MMD requires at least two units per group")
    if n_permutations < 0:
        raise ValueError("n_permutations must be nonnegative")
    if n_permutations and {u.unit_id for u in a.units} & {u.unit_id for u in b.units}:
        raise ValueError("Permutation groups must not contain the same units")
    check_pairwise_size(na + nb, na + nb, max_pairs)
    pooled = vstack([a.values, b.values], format="csr")
    squared = np.maximum(euclidean_distances(pooled, squared=True), 0)
    np.fill_diagonal(squared, 0)
    if bandwidth is None:
        positive = squared[np.triu_indices(na + nb, 1)]
        positive = positive[positive > 0]
        bandwidth = float(np.median(np.sqrt(positive))) if len(positive) else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    kernel = np.exp(-squared / (2 * bandwidth**2))

    def statistic(order: np.ndarray) -> float:
        ia, ib = order[:na], order[na:]
        aa, bb, ab = kernel[np.ix_(ia, ia)], kernel[np.ix_(ib, ib)], kernel[np.ix_(ia, ib)]
        if estimator == "biased":
            return float(max(0, aa.mean() + bb.mean() - 2 * ab.mean()))
        return float(
            (aa.sum() - np.trace(aa)) / (na * (na - 1))
            + (bb.sum() - np.trace(bb)) / (nb * (nb - 1))
            - 2 * ab.mean()
        )

    observed = statistic(np.arange(na + nb))
    p_value = None
    if n_permutations:
        rng = np.random.default_rng(random_state)
        extreme = sum(
            statistic(rng.permutation(na + nb)) >= observed for _ in range(n_permutations)
        )
        p_value = (extreme + 1) / (n_permutations + 1)
    return MMDResult(observed, bandwidth, estimator, na, nb, p_value, n_permutations, random_state)
