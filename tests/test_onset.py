import numpy as np

from cartpole_idk.generation import NormalOnset


def test_fixed_normal_onset():
    assert NormalOnset(50, 0, 0, 100).sample(np.random.default_rng(1)) == 50


def test_onset_clips():
    assert NormalOnset(200, 0, 0, 100).sample(np.random.default_rng(1)) == 100
    assert NormalOnset(-10, 0, 0, 100).sample(np.random.default_rng(1)) == 0
