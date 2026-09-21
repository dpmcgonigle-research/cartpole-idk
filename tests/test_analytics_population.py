from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cartpole_idk.analytics import (
    ClusterConfig,
    EmbeddingSet,
    cluster_purity,
    cluster_summary,
    cluster_units,
    evaluate_clusters,
    pairwise,
    population_mmd,
    reference_likeness,
    rolling_likeness,
    top_k_neighbors,
)
from cartpole_idk.idk.units import build_units


@pytest.fixture
def embedded(trajectory):
    units = build_units(
        [
            replace(
                trajectory,
                trajectory_id=str(i),
                metadata={
                    "perturbation_type": "nominal" if i < 6 else "failure",
                    "checkpoint_id": "checkpoint",
                },
            )
            for i in range(12)
        ]
    ).units
    a = np.linspace(0.85, 0.95, 6)
    values = csr_matrix(np.concatenate([np.column_stack([a, 1 - a]), np.column_stack([1 - a, a])]))
    return EmbeddingSet(units, values, "shared_basis", 1, 2)


@pytest.mark.parametrize("metric", ["idk", "idk-distance", "cosine", "js", "kl"])
def test_neighbor_direction_and_self_exclusion(embedded, metric):
    params = {"epsilon": 0.01} if metric == "kl" else {}
    query, reference = embedded.subset([0]), embedded.subset([0, 1, 11])
    result = top_k_neighbors(query, reference, metric=metric, k=10, metric_parameters=params)
    assert result.neighbors.reference_unit_id.tolist() == [
        reference.units[1].unit_id,
        reference.units[2].unit_id,
    ]
    assert result.neighbors["rank"].tolist() == [1, 2]
    assert result.scores.neighbor_count.tolist() == [2]
    assert result.scores.score.iloc[0] == pytest.approx(result.neighbors.metric_value.mean())


def test_same_trajectory_overlap_and_no_eligible(trajectory):
    units = build_units([trajectory], mode="window", window_length=2).units
    embedded = EmbeddingSet(units, csr_matrix([[1, 0], [0.9, 0.1]]), "basis", 1, 2)
    assert len(top_k_neighbors(embedded, embedded).neighbors) == 2
    for options in [{"exclude_same_trajectory": True}, {"max_overlap": 0.49}]:
        result = top_k_neighbors(embedded, embedded, **options)
        assert result.neighbors.empty
        assert result.scores.score.isna().all()
        assert result.scores.neighbor_count.tolist() == [0, 0]
    assert len(top_k_neighbors(embedded, embedded, max_overlap=0.5).neighbors) == 2
    assert len(top_k_neighbors(embedded, embedded, exclude_same_unit=False).neighbors) == 4


@pytest.mark.parametrize("metric", ["idk", "idk-distance"])
def test_separate_likeness_and_rolling(embedded, metric):
    query, nominal, failure = (
        embedded.subset([0]),
        embedded.subset([1, 2]),
        embedded.subset([10, 11]),
    )
    scores, results = reference_likeness(query, nominal, failure, metric=metric)
    assert scores.margin.iloc[0] > 0
    assert {"nominal_score", "failure_score"} <= set(scores)
    assert set(results) == {"nominal", "failure"}
    table = rolling_likeness(query, {"nominal": nominal, "failure": failure}, metrics=(metric,))
    assert len(table) == 2
    assert {"start_step", "end_step", "time_until_failure", "reference_group"} <= set(table)


@pytest.mark.parametrize("algorithm", ["dbscan", "hdbscan", "spectral", "dpgmm"])
def test_clustering_smoke(embedded, algorithm):
    config = ClusterConfig(
        algorithm=algorithm,
        eps=0.15,
        min_samples=2,
        min_cluster_size=3,
        n_clusters=2,
        svd_components=2,
        n_components=2,
    )
    result = cluster_units(embedded, config)
    assert result.labels.shape == (12,)
    assert result.assignments.unit_id.tolist() == [u.unit_id for u in embedded.units]
    assert len(np.unique(result.labels)) >= 2
    report = evaluate_clusters(result.assignments, result.distances)
    assert report["perturbation_type"]["purity"] == 1
    summary, composition = cluster_summary(result.assignments)
    assert summary["count"].sum() == 12
    assert set(composition.field) == {"perturbation_type", "checkpoint_id"}
    if algorithm == "dpgmm":
        assert result.coordinates.shape == (12, 2)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1)
        assert result.components["weights"].shape == (2,)
    if algorithm == "hdbscan":
        assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))


def test_precomputed_clustering_and_js(embedded):
    distances = pairwise(embedded, metric="idk-distance")
    config = ClusterConfig(algorithm="dbscan", eps=0.15, min_samples=2)
    first = cluster_units(embedded, config)
    second = cluster_units(embedded, config, precomputed=distances)
    np.testing.assert_array_equal(first.labels, second.labels)
    js = cluster_units(embedded, config.model_copy(update={"distance": "js"}))
    np.testing.assert_allclose(js.distances**2, pairwise(embedded, metric="js"), atol=1e-14)


def test_diagnostics_noise_and_undefined_silhouette(embedded):
    table = build_units([u.trajectory for u in embedded.units]).manifest()
    table["cluster"] = [-1] + [0] * 11
    report = evaluate_clusters(table, pairwise(embedded, metric="idk-distance"))
    assert report["n_noise"] == 1 and report["n_evaluated"] == 11
    assert report["silhouette"] is None
    assert cluster_purity(np.array([0, 0, 1, 1]), np.array(["a", "a", "a", "b"])) == 0.75


def test_mmd_identical_separated_and_deterministic_permutations(embedded):
    a, b = embedded.subset(list(range(6))), embedded.subset(list(range(6, 12)))
    assert population_mmd(a, a).statistic == pytest.approx(0, abs=1e-14)
    first = population_mmd(a, b, n_permutations=39, random_state=7)
    assert first.statistic > 0.5
    assert 0 < first.p_value <= 1
    assert first == population_mmd(a, b, n_permutations=39, random_state=7)
    unbiased = population_mmd(a, b, estimator="unbiased", bandwidth=0.5)
    assert unbiased.statistic > 0
    with pytest.raises(ValueError, match="two units"):
        population_mmd(a.subset([0]), b, estimator="unbiased")
    with pytest.raises(ValueError, match="same units"):
        population_mmd(a, a, n_permutations=2)
    with pytest.raises(ValueError, match="limit"):
        population_mmd(a, b, max_pairs=100)
