import numpy as np

from cartpole_idk.storage import Trajectory


def test_trajectory_roundtrip(tmp_path, trajectory):
    path = tmp_path / "traj.npz"
    trajectory.save(path)
    loaded = Trajectory.load(path)
    assert loaded.trajectory_id == trajectory.trajectory_id
    np.testing.assert_array_equal(loaded.true_observations, trajectory.true_observations)
    np.testing.assert_array_equal(loaded.executed_actions, trajectory.executed_actions)
    assert loaded.metadata == trajectory.metadata
