import json

import numpy as np
import pandas as pd
import pytest

from cartpole_idk.cli import generate, query
from cartpole_idk.idk import IDKExperimentConfig, fit_reference, trajectory_familiarity
from cartpole_idk.storage import TrajectoryStore
from cartpole_idk.training import TrainingConfig, load_checkpoint, train


@pytest.fixture
def trained_run(tmp_path):
    cfg = TrainingConfig(
        total_steps=8,
        max_episode_steps=4,
        hidden_sizes=(8,),
        batch_size=2,
        learning_starts=2,
        target_update_every=2,
        eval_every=4,
        eval_episodes=1,
        checkpoint_every=4,
    )
    return train(tmp_path / "run", cfg)


def test_training_writes_metrics_and_loadable_checkpoints(trained_run):
    config = json.loads((trained_run / "config.json").read_text())
    assert config["total_steps"] == 8
    metrics = pd.read_csv(trained_run / "training_metrics.csv")
    assert metrics["step"].tolist() == [4, 8]
    assert metrics["length"].tolist() == [4, 4]
    assert np.isfinite(metrics["loss"]).all()
    evaluation = pd.read_csv(trained_run / "evaluation_metrics.csv")
    assert evaluation["step"].tolist() == [4, 8]
    for step in (4, 8):
        agent, payload = load_checkpoint(trained_run / "checkpoints" / f"step_{step:09d}.pt")
        assert payload["step"] == step
        assert agent.act(np.zeros(4, dtype=np.float32)) in (0, 1)


def test_generate_query_and_evaluate_workflow(trained_run, tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    checkpoint = trained_run / "checkpoints" / "step_000000008.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cartpole-generate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(dataset),
            "--episodes",
            "2",
            "--max-episode-steps",
            "4",
        ],
    )
    generate.main(standalone_mode=False)
    store = TrajectoryStore(dataset)
    ids = store.manifest()["trajectory_id"].tolist()
    assert len(ids) == len(set(ids)) == 2
    report = json.loads((dataset / "generation_report.json").read_text())
    assert report["return_statistics"] == {
        "mean": 4,
        "std_dev": 0,
        "min": 4,
        "max": 4,
        "q1": 4,
        "median": 4,
        "q3": 4,
    }
    assert report["episodes"] == 2
    assert [row["trajectory_id"] for row in report["generated"]] == ids
    trajectories = [store.get(tid) for tid in ids]
    for trajectory in trajectories:
        assert trajectory.length == 4
        assert trajectory.truncated[-1]
        assert trajectory.true_observations.shape == (5, 4)
        np.testing.assert_array_equal(trajectory.agent_observations, trajectory.true_observations)
        np.testing.assert_array_equal(trajectory.commanded_actions, trajectory.executed_actions)

    reference = fit_reference(store, ids, IDKExperimentConfig(psi=2, t=4))
    scores = trajectory_familiarity(reference, trajectories, k=5)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="k must be >= 1"):
        trajectory_familiarity(reference, trajectories, k=0)

    monkeypatch.setattr("sys.argv", ["cartpole-query", str(dataset), "--min-return", "4"])
    query.main(standalone_mode=False)
    output = capsys.readouterr().out
    assert all(tid in output for tid in ids)
