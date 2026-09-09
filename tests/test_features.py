import numpy as np

from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.idk.features import build_sequence_batch


def test_state_representation(trajectory):
    batch = build_sequence_batch([trajectory], IDKExperimentConfig(representation="state"))
    assert batch.values.shape == (4, 4)


def test_transition_representation(trajectory):
    batch = build_sequence_batch([trajectory], IDKExperimentConfig(representation="transition"))
    assert batch.values.shape == (3, 8)


def test_state_action_representation(trajectory):
    batch = build_sequence_batch([trajectory], IDKExperimentConfig(representation="state_action"))
    assert batch.values.shape == (3, 6)
    np.testing.assert_array_equal(batch.values[0, -2:], [1.0, 0.0])


def test_state_action_next_state_representation(trajectory):
    batch = build_sequence_batch(
        [trajectory], IDKExperimentConfig(representation="state_action_next_state")
    )
    assert batch.values.shape == (3, 10)
