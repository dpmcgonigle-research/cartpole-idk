from dataclasses import replace

import numpy as np
import pytest

from cartpole_idk.analytics import build_units, fit_embeddings, pairwise
from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.idk.features import build_sequence_batch


def test_windows_provenance_and_boundaries(trajectory):
    second = replace(trajectory, trajectory_id="other", metadata={"perturbation_onset": 1})
    collection = build_units([trajectory, second], mode="window", window_length=2)
    assert len(collection.units) == 4
    assert [(u.start_step, u.end_step) for u in collection.units] == [(0, 2), (1, 3)] * 2
    for unit in collection.units:
        assert unit.raw_length == unit.trajectory.length == 2
        assert len(unit.trajectory.true_observations) == 3
        assert unit.metadata["time_until_failure"] == 3 - unit.end_step
        assert unit.metadata["episode_return"] == 3
    assert collection.units[-1].metadata["time_since_perturbation"] == 2
    assert len(build_units([trajectory], mode="window", window_length=2, stride=2).units) == 1
    short = build_units([trajectory], mode="window", window_length=4)
    assert not short.units and short.skipped[0]["reason"] == "too_short"
    with pytest.raises(ValueError, match="boundary"):
        build_units([replace(trajectory, terminated=np.array([True, False, True]))])


@pytest.mark.parametrize(
    "representation,samples,width",
    [
        ("state", 4, 4),
        ("transition", 3, 8),
        ("state_action", 3, 6),
        ("state_action_next_state", 3, 10),
        ("window", 3, 8),
    ],
)
def test_reuse_all_representations(trajectory, representation, samples, width):
    config = IDKExperimentConfig(representation=representation, window_length=2, psi=2, t=3)
    units = build_units([trajectory])
    batch = build_sequence_batch([units.units[0].trajectory], config)
    assert batch.values.shape == (samples, width)
    basis = fit_embeddings(units, config)
    assert basis.fitting.values.shape == (1, 6)
    np.testing.assert_allclose(
        basis.fitting.values.toarray(), basis.transform(units).values.toarray()
    )


def test_empty_representation_is_reported(trajectory):
    short = replace(
        trajectory,
        trajectory_id="short",
        true_observations=trajectory.true_observations[:2],
        agent_observations=trajectory.agent_observations[:2],
        commanded_actions=trajectory.commanded_actions[:1],
        executed_actions=trajectory.executed_actions[:1],
        rewards=trajectory.rewards[:1],
        terminated=trajectory.terminated[:1],
        truncated=trajectory.truncated[:1],
    )
    cfg = IDKExperimentConfig(representation="window", window_length=3, psi=2, t=2)
    fitted = fit_embeddings(build_units([trajectory, short]), cfg)
    assert len(fitted.fitting.units) == 1
    assert fitted.fitting.skipped[0]["reason"] == "empty_representation"
    with pytest.raises(ValueError, match="empty representations"):
        fitted.transform(build_units([short]))


def test_fit_transform_does_not_leak_labels_or_query_values(trajectory):
    config = IDKExperimentConfig(psi=2, t=4)
    fitted = fit_embeddings(build_units([trajectory]), config)
    mean = fitted.reference.scaler.mean_.copy()
    centers = fitted.reference.model.point_kernel.basis_.centers.copy()
    query = replace(
        trajectory,
        trajectory_id="query",
        true_observations=trajectory.true_observations + 1000,
        metadata={"success": False, "perturbation_type": "novel"},
    )
    embedded = fitted.transform(build_units([query]))
    np.testing.assert_array_equal(mean, fitted.reference.scaler.mean_)
    np.testing.assert_array_equal(centers, fitted.reference.model.point_kernel.basis_.centers)
    assert embedded.values.nnz == 0
    relabeled = replace(trajectory, metadata={"success": True, "group": "failure"})
    refit = fit_embeddings(build_units([relabeled]), config)
    np.testing.assert_array_equal(centers, refit.reference.model.point_kernel.basis_.centers)
    with pytest.raises(ValueError, match="same fitted basis"):
        pairwise(fitted.fitting, refit.fitting)


def test_prepared_prefix_is_censored(trajectory):
    prefix = replace(
        trajectory,
        terminated=np.zeros(3, dtype=bool),
        metadata={
            "source_length": 50,
            "source_return": 50,
            "source_trajectory_id": "original",
            "segment_start": 10,
            "perturbation_onset": -5,
        },
    )
    unit = build_units([prefix]).units[0]
    assert unit.source_key == ("", "original")
    assert unit.metadata["source_start_step"] == 10
    assert unit.metadata["source_end_step"] == 13
    assert unit.metadata["time_until_termination"] is None
    assert unit.metadata["time_until_failure"] is None
    assert unit.metadata["episode_return"] == 50
