import numpy as np
import pytest

from cartpole_idk.storage import Trajectory


@pytest.fixture
def trajectory():
    true_obs = np.array(
        [
            [0.0, 0.0, 0.01, 0.0],
            [0.01, 0.1, 0.02, 0.2],
            [0.02, 0.2, 0.03, 0.3],
            [0.03, 0.3, 0.04, 0.4],
        ],
        dtype=np.float32,
    )
    return Trajectory(
        trajectory_id="traj_test",
        true_observations=true_obs,
        agent_observations=true_obs.copy(),
        commanded_actions=np.array([0, 1, 0], dtype=np.int8),
        executed_actions=np.array([0, 1, 0], dtype=np.int8),
        rewards=np.ones(3, dtype=np.float32),
        terminated=np.array([False, False, True]),
        truncated=np.array([False, False, False]),
        metadata={
            "checkpoint_id": "step_100",
            "perturbation_type": "none",
            "perturbation_onset": None,
        },
    )
