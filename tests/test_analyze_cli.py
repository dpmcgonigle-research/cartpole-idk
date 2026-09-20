import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy.sparse import load_npz

from cartpole_idk.cli.analyze import main
from cartpole_idk.storage import TrajectoryStore


@pytest.fixture
def analysis_dataset(tmp_path, trajectory):
    store = TrajectoryStore(tmp_path / "dataset")
    for i in range(8):
        store.add(
            replace(
                trajectory,
                trajectory_id=f"traj_{i}",
                true_observations=trajectory.true_observations + i * 0.01,
                metadata={"perturbation_type": "none" if i < 4 else "flip"},
            )
        )
    for name, ids in {
        "fit": [0, 1, 4, 5],
        "test": [2, 3, 6, 7],
        "nominal": [0, 1],
        "failure": [4, 5],
        "a": [2, 3],
        "b": [6, 7],
    }.items():
        (tmp_path / f"{name}.txt").write_text("\n".join(f"traj_{i}" for i in ids))
    return store


@pytest.mark.parametrize(
    "command,extra",
    [
        ("pairwise", ["--metric", "js"]),
        ("neighbors", ["--metric", "idk-distance"]),
        ("rolling", ["--query-id", "traj_2"]),
        ("cluster", ["--algorithm", "hdbscan", "--min-cluster-size", "2"]),
        ("cluster", ["--algorithm", "dbscan", "--min-samples", "2"]),
        ("cluster", ["--algorithm", "spectral", "--n-clusters", "2"]),
        ("cluster", ["--algorithm", "dpgmm", "--svd-components", "2", "--n-components", "2"]),
        ("population", ["--permutations", "9"]),
    ],
)
def test_cli_runs(analysis_dataset, tmp_path, command, extra):
    output = tmp_path / "result"
    args = [
        command,
        str(analysis_dataset.root),
        "--output",
        str(output),
        "--fit-ids-file",
        str(tmp_path / "fit.txt"),
        "--analysis-ids-file",
        str(tmp_path / "test.txt"),
        "--psi",
        "2",
        "--t",
        "4",
        "--window-length",
        "2",
        "--representation",
        "transition",
    ]
    if command != "rolling":
        args += ["--mode", "window"]
    if command == "population":
        args += [
            "--group-a-ids-file",
            str(tmp_path / "a.txt"),
            "--group-b-ids-file",
            str(tmp_path / "b.txt"),
        ]
    if command == "rolling":
        args += [
            "--nominal-reference-ids-file",
            str(tmp_path / "nominal.txt"),
            "--failure-reference-ids-file",
            str(tmp_path / "failure.txt"),
        ]
    result = CliRunner().invoke(main, args + extra)
    assert result.exit_code == 0, result.output
    config = json.loads((output / "config.json").read_text())
    assert config["trajectory_roles"]["fit"] == ["traj_0", "traj_1", "traj_4", "traj_5"]
    units = pd.read_parquet(output / "units.parquet")
    matrix = load_npz(output / "embeddings.npz")
    assert matrix.shape == (len(units), 8)
    assert units.unit_id.is_unique
    assert (output / "basis.npz").exists()
    if command == "pairwise":
        axes = json.loads((output / "pairwise_axes.json").read_text())
        assert axes["rows"] == config["unit_roles"]["analysis"]
        assert np.load(output / "pairwise.npy").shape == (8, 8)
    if command == "rolling":
        scores = pd.read_parquet(output / "rolling.parquet")
        assert {"nominal_score", "failure_score", "margin", "time_until_failure"} <= set(scores)
        assert scores.start_step.tolist() == [0, 1]
    if command == "population":
        result = json.loads((output / "population.json").read_text())
        assert result["n_a"] == result["n_b"] == 4
        assert result["n_permutations"] == 9


def test_whole_defaults_and_output_protection(analysis_dataset, tmp_path):
    args = [
        "pairwise",
        str(analysis_dataset.root),
        "--output",
        str(tmp_path / "out"),
        "--psi",
        "2",
        "--t",
        "3",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert np.load(tmp_path / "out/pairwise.npy").shape == (8, 8)
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert "Output must be a new or empty directory" in result.output


def test_parser_and_invalid_ids(analysis_dataset, tmp_path):
    (tmp_path / "bad.txt").write_text("not_in_dataset")
    result = CliRunner().invoke(
        main,
        [
            "pairwise",
            str(analysis_dataset.root),
            "--fit-ids-file",
            str(tmp_path / "bad.txt"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "Unknown trajectory IDs" in result.output
    assert not (tmp_path / "out").exists()


def test_cached_fit_population_with_short_representation(tmp_path, trajectory):
    from cartpole_idk.analytics.workflow import AnalysisConfig, AnalysisRun
    from cartpole_idk.idk import IDKExperimentConfig

    store = TrajectoryStore(tmp_path / "data")
    store.add(trajectory)
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
    store.add(short)
    config = AnalysisConfig(
        store.root,
        tmp_path / "out",
        idk=IDKExperimentConfig(representation="window", window_length=3, psi=2, t=2),
    )
    run = AnalysisRun(config, [trajectory.trajectory_id, "short"])
    result = run.embed("analysis", [trajectory.trajectory_id, "short"])
    assert len(result.units) == 1
    assert result.skipped[0]["reason"] == "empty_representation"


@pytest.mark.parametrize(
    "command", [None, "pairwise", "neighbors", "rolling", "cluster", "population"]
)
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_click_help(command, flag):
    import click

    assert isinstance(main, click.Group)
    args = [command, flag] if command else [flag]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_cluster_defaults_and_space_separated_categories(monkeypatch):
    from unittest.mock import Mock

    from cartpole_idk.cli import analyze

    execute = Mock()
    monkeypatch.setattr(analyze, "execute_analysis", execute)
    for categories in [[], ["--categorical", "success", "perturbation_type"]]:
        result = CliRunner().invoke(
            main,
            [
                "cluster",
                "dataset",
                "--output",
                "out",
                *categories,
                "--random-state",
                "7",
            ],
        )
        assert result.exit_code == 0, result.output
        args, kwargs = execute.call_args
        assert args[0] == "cluster"
        assert args[1].idk.random_state == 7
        options = kwargs["options"]
        assert options["cluster_config"].algorithm == "hdbscan"
        assert options["categorical"] == (
            ("success", "perturbation_type")
            if categories
            else ("perturbation_type", "checkpoint_id")
        )


@pytest.mark.parametrize(
    "args,message",
    [
        (["rolling", "data", "--output", "out"], "--query-id"),
        (["population", "data", "--output", "out"], "--group-a-ids-file"),
        (["pairwise", "data", "--output", "out", "--metric", "invalid"], "Invalid value"),
        (["pairwise", "data", "--output", "out", "--metric", "kl"], "--epsilon > 0"),
        (
            ["cluster", "data", "--output", "out", "--categorical", "--eps", "0.2"],
            "requires an argument",
        ),
    ],
)
def test_click_usage_errors(args, message):
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert message in result.output
