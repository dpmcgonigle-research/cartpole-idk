from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from cartpole_idk.cli import embed, fit, generate, prepare, query, replay, train


@pytest.mark.parametrize("module", [train, generate, prepare, query, replay, fit, embed])
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help(module, flag):
    assert isinstance(module.main, click.Command)
    result = CliRunner().invoke(module.main, [flag])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("sizes", [[], ["--hidden-sizes", "64", "32", "16"]])
def test_train_options(monkeypatch, sizes):
    mocked = Mock()
    monkeypatch.setattr(train, "train", mocked)
    result = CliRunner().invoke(train.main, ["--run-dir", "run", *sizes, "--seed", "7"])
    assert result.exit_code == 0, result.output
    args, kwargs = mocked.call_args
    assert args[0] == "run"
    assert args[1].hidden_sizes == ((64, 32, 16) if sizes else (128, 128))
    assert args[1].seed == 7
    assert args[1].total_steps == 200000
    assert kwargs == {"device": "cpu"}


@pytest.mark.parametrize(
    "perturbation, cls",
    [
        ("none", generate.Perturbation),
        ("action-delay", generate.ActionDelay),
        ("action-flip", generate.ActionFlip),
        ("observation-bias", generate.ObservationBias),
    ],
)
def test_generate_options(monkeypatch, perturbation, cls):
    mocked = Mock()
    monkeypatch.setattr(generate, "generate_dataset", mocked)
    result = CliRunner().invoke(
        generate.main,
        [
            "--checkpoint",
            "model.pt",
            "--output",
            "out",
            "--perturbation",
            perturbation,
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = mocked.call_args.kwargs
    assert isinstance(kwargs["perturbation"], cls)
    assert kwargs["episodes"] == 100
    assert kwargs["max_episode_steps"] == 500
    assert kwargs["seed"] == 1000


def test_replay_options(monkeypatch):
    store = Mock()
    mocked = Mock()
    monkeypatch.setattr(replay, "TrajectoryStore", Mock(return_value=store))
    monkeypatch.setattr(replay, "replay_trajectory", mocked)
    result = CliRunner().invoke(replay.main, ["data", "--trajectory", "tid", "--fps", "25"])
    assert result.exit_code == 0, result.output
    store.get.assert_called_once_with("tid")
    mocked.assert_called_once_with(store.get.return_value, fps=25)


def test_prepare_usage_error():
    result = CliRunner().invoke(prepare.main, ["data", "--output", "out", "--episode-length", "0"])
    assert result.exit_code == 2
    assert "min_episode_length" in result.output


def test_list_option_missing_value():
    result = CliRunner().invoke(train.main, ["--run-dir", "run", "--hidden-sizes", "--seed", "3"])
    assert result.exit_code == 2
    assert "requires an argument" in result.output
