from cartpole_idk.storage import TrajectoryStore


def test_store_add_get_query(tmp_path, trajectory):
    store = TrajectoryStore(tmp_path)
    store.add(trajectory)
    assert store.get("traj_test").length == 3
    rows = store.query(checkpoint_id="step_100", perturbation_type="none")
    assert len(rows) == 1
    assert rows.iloc[0]["trajectory_id"] == "traj_test"


def test_query_return_range(tmp_path, trajectory):
    store = TrajectoryStore(tmp_path)
    store.add(trajectory)
    assert len(store.query(min_return=2, max_return=4)) == 1
    assert len(store.query(max_return=2)) == 0
