import json
import shutil
from unittest.mock import Mock

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pyidk import IsolationDistributionalKernel, IsolationKernel, Standardizer

from cartpole_idk.cli import analyze, embed, fit
from cartpole_idk.storage import TrajectoryStore


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
def test_analysis_uses_only_embedding_artifacts(
    embedding_artifacts, tmp_path, monkeypatch, command, extra
):
    shutil.rmtree(tmp_path / "dataset")
    shutil.rmtree(tmp_path / "fit")

    def forbidden(*args, **kwargs):
        pytest.fail("Analytics must never fit, embed, or read raw trajectories")

    for cls in (IsolationKernel, IsolationDistributionalKernel, Standardizer):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "transform", forbidden)
    monkeypatch.setattr(TrajectoryStore, "get", forbidden)
    from cartpole_idk.artifacts import FitArtifact

    monkeypatch.setattr(FitArtifact, "load", forbidden)
    path = embedding_artifacts["a" if command == "population" else "test"]
    output = tmp_path / "result"
    args = [command, str(path), "--output", str(output), *extra]
    if command == "population":
        args += ["--group-b", str(embedding_artifacts["b"])]
    if command == "rolling":
        args += [
            "--nominal-reference",
            str(embedding_artifacts["nominal"]),
            "--failure-reference",
            str(embedding_artifacts["failure"]),
        ]
    result = CliRunner().invoke(analyze.main, args)
    assert result.exit_code == 0, result.output
    config = json.loads((output / "config.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    assert config["command"] == command
    assert metadata["inputs"]["analysis"]["artifact_id"]
    units = pd.read_parquet(output / "units.parquet")
    assert units.unit_id.is_unique
    assert not (output / "embeddings.npz").exists()
    assert not (output / "basis.npz").exists()
    if command == "pairwise":
        axes = json.loads((output / "pairwise_axes.json").read_text())
        assert axes["rows"] == metadata["unit_roles"]["analysis"]
        assert np.load(output / "pairwise.npy").shape == (8, 8)
    if command == "rolling":
        scores = pd.read_parquet(output / "rolling.parquet")
        assert {"nominal_score", "failure_score", "margin"} <= set(scores)
        assert scores.start_step.tolist() == [0, 1]
    if command == "population":
        result_mmd = json.loads((output / "population.json").read_text())
        assert result_mmd["n_a"] == result_mmd["n_b"] == 4
        assert result_mmd["n_permutations"] == 9


def test_fit_embed_cli_and_no_refit(pipeline_dataset, tmp_path, monkeypatch):
    runner = CliRunner()
    fit_path, embed_path = tmp_path / "fit_cli", tmp_path / "embed_cli"
    ids = tmp_path / "ids.txt"
    ids.write_text("traj_0\ntraj_1\n")
    result = runner.invoke(
        fit.main,
        [
            str(pipeline_dataset.root),
            "--ids-file",
            str(ids),
            "--output",
            str(fit_path),
            "--representation",
            "transition",
            "--psi",
            "2",
            "--t",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output

    def forbidden(*args, **kwargs):
        pytest.fail("cartpole-embed refitted")

    for cls in (IsolationKernel, IsolationDistributionalKernel, Standardizer):
        monkeypatch.setattr(cls, "fit", forbidden)
    result = runner.invoke(
        embed.main,
        [
            "--fit",
            str(fit_path),
            "--dataset",
            str(pipeline_dataset.root),
            "--ids-file",
            str(ids),
            "--output",
            str(embed_path),
            "--mode",
            "window",
            "--window-length",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(embed_path / "units.parquet")) == 4
    result = runner.invoke(
        analyze.main, ["pairwise", str(embed_path), "--output", str(tmp_path / "analysis")]
    )
    assert result.exit_code == 0, result.output


def test_query_reference_pairwise_and_output_protection(embedding_artifacts, tmp_path):
    args = [
        "pairwise",
        str(embedding_artifacts["test"]),
        "--reference",
        str(embedding_artifacts["nominal"]),
        "--output",
        str(tmp_path / "out"),
    ]
    result = CliRunner().invoke(analyze.main, args)
    assert result.exit_code == 0, result.output
    assert np.load(tmp_path / "out/pairwise.npy").shape == (8, 4)
    assert CliRunner().invoke(analyze.main, args).exit_code == 2


def test_rolling_requires_existing_windows(embedding_artifacts, tmp_path):
    result = CliRunner().invoke(
        analyze.main,
        [
            "rolling",
            str(embedding_artifacts["whole"]),
            "--query-id",
            "traj_2",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2 and "existing window embedding" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "command", [None, "pairwise", "neighbors", "rolling", "cluster", "population"]
)
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_click_help(command, flag):
    assert isinstance(analyze.main, click.Group)
    result = CliRunner().invoke(analyze.main, [command, flag] if command else [flag])
    assert result.exit_code == 0, result.output
    assert "--psi" not in result.output and "--fit-ids-file" not in result.output


def test_cluster_defaults_and_space_separated_categories(monkeypatch):
    execute = Mock()
    monkeypatch.setattr(analyze, "execute_analysis", execute)
    result = CliRunner().invoke(
        analyze.main,
        [
            "cluster",
            "embeddings",
            "--output",
            "out",
            "--categorical",
            "success",
            "perturbation_type",
            "--random-state",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    config = execute.call_args.args[0]
    assert config.cluster.algorithm == "hdbscan"
    assert config.cluster.categorical == ("success", "perturbation_type")
    assert config.cluster.random_state == 7


@pytest.mark.parametrize(
    "args,message",
    [
        (["rolling", "data", "--output", "out"], "--query-id"),
        (["population", "data", "--output", "out"], "--group-b"),
        (["pairwise", "data", "--output", "out", "--metric", "invalid"], "Invalid value"),
        (["pairwise", "data", "--output", "out", "--metric", "kl"], "--epsilon > 0"),
        (["pairwise", "data", "--output", "out", "--psi", "2"], "No such option"),
    ],
)
def test_click_usage_errors(args, message):
    result = CliRunner().invoke(analyze.main, args)
    assert result.exit_code == 2 and message in result.output
