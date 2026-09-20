import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cartpole_idk.analytics.metrics import (
    METRICS,
    pairwise_cosine,
    pairwise_idk_distance,
    pairwise_idk_similarity,
    pairwise_js_divergence,
    pairwise_kl_divergence,
)


@pytest.fixture
def occupancies():
    return csr_matrix([[0.5, 0, 0.25, 0.25], [0, 1, 0.5, 0], [0, 0, 0, 0]])


def test_native_similarity(occupancies):
    values = pairwise_idk_similarity(occupancies, t=2)
    np.testing.assert_allclose(values, (occupancies @ occupancies.T).toarray() / 2)
    np.testing.assert_allclose(values, values.T)
    assert values[0, 0] == 0.1875
    np.testing.assert_allclose(
        pairwise_idk_similarity(occupancies[:1], occupancies[1:], t=2), values[:1, 1:]
    )


def test_idk_distance(occupancies):
    actual = pairwise_idk_distance(occupancies, t=2)
    dense = occupancies.toarray()
    expected = np.linalg.norm(dense[:, None] - dense[None, :], axis=-1) / np.sqrt(2)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual.diagonal(), 0)
    np.testing.assert_allclose(actual, actual.T)


def test_distance_roundoff_is_clipped(monkeypatch):
    import cartpole_idk.analytics.metrics as module

    monkeypatch.setattr(module, "pairwise_similarity", lambda *a, **kw: np.array([[1 + 1e-15]]))
    result = pairwise_idk_distance(csr_matrix([[1, 0]]), csr_matrix([[1, 0]]), t=1)
    np.testing.assert_array_equal(result, [[0]])


def test_cosine_zero_and_orthogonal():
    values = pairwise_cosine(csr_matrix([[0.5, 0], [0, 1], [0, 0]]))
    np.testing.assert_array_equal(values, np.diag([1, 1, 0]))


def test_js_preserves_outside_mass():
    x = csr_matrix([[1, 0], [0, 0], [0.5, 0], [0, 1]])
    result = pairwise_js_divergence(x, t=1)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(result.diagonal(), 0, atol=1e-15)
    assert np.isfinite(result).all()
    assert result[0, 1] == pytest.approx(np.log(2))
    assert result[0, 3] == pytest.approx(np.log(2))
    assert result[0, 2] == pytest.approx(0.21576155433883565)
    # Two partitions: one equal, the other disjoint.
    both = csr_matrix([[1, 0, 1, 0], [1, 0, 0, 1]])
    assert pairwise_js_divergence(both, t=2)[0, 1] == pytest.approx(np.log(2) / 2)


def test_kl_direction_smoothing_and_zeros(occupancies):
    result = pairwise_kl_divergence(occupancies, t=2, epsilon=0.01)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.diagonal(), 0, atol=1e-15)
    assert result[0, 1] != pytest.approx(result[1, 0])
    with pytest.raises(ValueError, match="epsilon"):
        pairwise_kl_divergence(occupancies, t=2, epsilon=0)


@pytest.mark.parametrize("values", [[[1.1, 0]], [[-0.1, 0.5]], [[np.nan, 0]]])
def test_invalid_occupancy(values):
    with pytest.raises(ValueError):
        pairwise_js_divergence(csr_matrix(values), t=1)


def test_size_guard_and_metric_direction(occupancies):
    with pytest.raises(ValueError, match="limit"):
        pairwise_idk_similarity(occupancies, t=2, max_pairs=8)
    assert {name: spec.direction for name, spec in METRICS.items()} == {
        "idk": "higher",
        "idk-distance": "lower",
        "cosine": "higher",
        "js": "lower",
        "kl": "lower",
    }
