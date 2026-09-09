import numpy as np

from cartpole_idk.generation import ActionDelay, ActionFlip, NormalOnset, ObservationBias


def test_action_delay():
    p = ActionDelay(NormalOnset(1, 0), delay_steps=2)
    p.reset(np.random.default_rng(1), 20)
    commands = [0, 1, 1, 0, 1, 0, 0]
    executed = [p.executed_action(a, i) for i, a in enumerate(commands)]
    # Hold the onset command while the queue fills, then execute two steps behind.
    assert executed == [0, 1, 1, 1, 1, 0, 1]


def test_action_flip_probability_one():
    p = ActionFlip(NormalOnset(2, 0), probability=1.0)
    p.reset(np.random.default_rng(1), 20)
    assert p.executed_action(0, 1) == 0
    assert p.executed_action(0, 2) == 1


def test_observation_bias_preserves_true_input():
    p = ObservationBias(NormalOnset(0, 0), feature="pole_angle", bias=0.5)
    p.reset(np.random.default_rng(1), 20)
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    changed = p.agent_observation(obs, 0)
    assert np.isclose(changed[2], 3.5)
    assert np.isclose(obs[2], 3.0)
