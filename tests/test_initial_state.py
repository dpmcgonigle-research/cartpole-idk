import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from cartpole_idk.analytics.workflow import execute_analysis
from cartpole_idk.artifacts import EmbeddingArtifact, FitArtifact
from cartpole_idk.cli import generate
from cartpole_idk.generation import (
    NormalOnset,
    ObservationBias,
    generate_dataset,
    generator,
    sample_initial_state,
)
from cartpole_idk.idk.pipeline import embed_dataset, fit_dataset
from cartpole_idk.model import (
    AnalysisConfig,
    Distribution,
    EmbedConfig,
    FitConfig,
    GenerationReport,
    IDKConfig,
    InitialStateConfig,
    NormalInitialState,
    UniformInitialState,
    WindowConfig,
)
from cartpole_idk.preparation import prepare_dataset
from cartpole_idk.training.checkpoint import save_checkpoint
from cartpole_idk.training.dqn import DQNAgent


@pytest.mark.parametrize("distribution", list(Distribution))
def test_cli_distribution_enum_and_all_parameters(monkeypatch, distribution):
    mocked = Mock()
    monkeypatch.setattr(generate, "generate_dataset", mocked)
    args = [
        "--checkpoint",
        "model.pt",
        "--output",
        "data",
        "--initial-distribution",
        distribution.value,
    ]
    expected = {}
    for i, variable in enumerate(("x", "x_dot", "theta", "theta_dot")):
        values = (
            {"min": -i - 1.0, "max": i + 2.0}
            if distribution is Distribution.UNIFORM
            else {
                "mean": i * 0.1,
                "std": (i + 1) * 0.03,
            }
        )
        for suffix, value in values.items():
            args += [f"--initial-{variable.replace('_', '-')}-{suffix}", str(value)]
            expected[f"{variable}_{suffix}"] = value
    result = CliRunner().invoke(generate.main, args)
    assert result.exit_code == 0, result.output
    config = mocked.call_args.kwargs["initial_state"]
    assert config.distribution is distribution
    assert config.parameters.model_dump() == expected
    assert config.model_dump(mode="json") == {
        "distribution": distribution.value,
        "parameters": expected,
    }


def test_default_cli_and_inactive_parameters(monkeypatch):
    mocked = Mock()
    monkeypatch.setattr(generate, "generate_dataset", mocked)
    result = CliRunner().invoke(generate.main, ["--checkpoint", "model.pt", "--output", "data"])
    assert result.exit_code == 0, result.output
    assert mocked.call_args.kwargs["initial_state"] == InitialStateConfig()
    result = CliRunner().invoke(
        generate.main,
        [
            "--checkpoint",
            "model.pt",
            "--output",
            "data",
            "--initial-distribution",
            "normal",
            "--initial-x-min",
            "10",
            "--initial-x-max",
            "-10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert isinstance(mocked.call_args.kwargs["initial_state"].parameters, NormalInitialState)


@pytest.mark.parametrize(
    "args",
    [
        ["--initial-distribution", "unknown"],
        ["--initial-x-min", "1", "--initial-x-max", "1"],
        ["--initial-distribution", "normal", "--initial-theta-std", "-0.1"],
        ["--initial-x-min", "nan"],
    ],
)
def test_cli_invalid_config_is_usage_error(monkeypatch, args):
    mocked = Mock()
    monkeypatch.setattr(generate, "generate_dataset", mocked)
    result = CliRunner().invoke(
        generate.main, ["--checkpoint", "model.pt", "--output", "data", *args]
    )
    assert result.exit_code == 2, result.output
    mocked.assert_not_called()


def test_uniform_independent_bounds_and_rng():
    low = np.array([-1, 0.1, -0.2, 2])
    high = np.array([-0.5, 0.2, -0.1, 3])
    config = InitialStateConfig(
        parameters=UniformInitialState(
            x_min=low[0],
            x_max=high[0],
            x_dot_min=low[1],
            x_dot_max=high[1],
            theta_min=low[2],
            theta_max=high[2],
            theta_dot_min=low[3],
            theta_dot_max=high[3],
        )
    )
    rng = np.random.default_rng(42)
    expected_rng = np.random.default_rng(42)
    for _ in range(100):
        value = sample_initial_state(config, rng)
        assert value.shape == (4,)
        assert (value >= low).all() and (value < high).all()
        np.testing.assert_array_equal(value, expected_rng.uniform(low, high))


def test_normal_independent_means_and_stds():
    means = np.array([1, -2, 0.5, 4])
    stds = np.array([0.1, 0.2, 0, 0.4])
    config = InitialStateConfig(
        distribution=Distribution.NORMAL,
        parameters=NormalInitialState(
            x_mean=means[0],
            x_std=stds[0],
            x_dot_mean=means[1],
            x_dot_std=stds[1],
            theta_mean=means[2],
            theta_std=stds[2],
            theta_dot_mean=means[3],
            theta_dot_std=stds[3],
        ),
    )
    actual_rng, expected_rng = np.random.default_rng(17), np.random.default_rng(17)
    for _ in range(100):
        np.testing.assert_array_equal(
            sample_initial_state(config, actual_rng), expected_rng.normal(means, stds)
        )


def test_normal_is_not_clipped_to_environment_bounds():
    config = InitialStateConfig(
        distribution=Distribution.NORMAL,
        parameters=NormalInitialState(
            x_mean=10,
            x_std=0,
            x_dot_mean=-20,
            x_dot_std=0,
            theta_mean=5,
            theta_std=0,
            theta_dot_mean=-40,
            theta_dot_std=0,
        ),
    )
    np.testing.assert_array_equal(
        sample_initial_state(config, np.random.default_rng(0)), [10, -20, 5, -40]
    )


@pytest.mark.parametrize(
    "config",
    [
        InitialStateConfig(),
        InitialStateConfig(distribution=Distribution.NORMAL, parameters=NormalInitialState()),
    ],
)
def test_sampling_seeds(config):
    a = sample_initial_state(config, np.random.default_rng(42))
    np.testing.assert_array_equal(a, sample_initial_state(config, np.random.default_rng(42)))
    assert not np.array_equal(a, sample_initial_state(config, np.random.default_rng(43)))
    assert InitialStateConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("variable", ["x", "x_dot", "theta", "theta_dot"])
@pytest.mark.parametrize("low,high", [(1, 1), (2, 1)])
def test_invalid_uniform_bounds(variable, low, high):
    with pytest.raises(ValidationError, match="min must be less than max"):
        UniformInitialState(**{f"{variable}_min": low, f"{variable}_max": high})


@pytest.mark.parametrize("variable", ["x", "x_dot", "theta", "theta_dot"])
def test_negative_normal_std(variable):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        NormalInitialState(**{f"{variable}_std": -1})


def test_mismatched_distribution_parameters():
    with pytest.raises(ValidationError, match="incompatible parameters"):
        InitialStateConfig(distribution=Distribution.NORMAL, parameters=UniformInitialState())


@pytest.fixture
def checkpoint(tmp_path):
    agent = DQNAgent(4, 2, (8,), learning_rate=0.001, gamma=0.99, seed=42)
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, path, step=0, metadata={})
    return path


@pytest.mark.parametrize("distribution", list(Distribution))
@pytest.mark.parametrize("biased", [False, True])
def test_generator_installs_sampled_state_and_preserves_time_limit(
    tmp_path,
    checkpoint,
    monkeypatch,
    distribution,
    biased,
):
    config = (
        InitialStateConfig()
        if distribution is Distribution.UNIFORM
        else InitialStateConfig(
            distribution=distribution,
            parameters=NormalInitialState(
                x_mean=0.1,
                x_std=0,
                x_dot_mean=-0.03,
                x_dot_std=0,
                theta_mean=0.01,
                theta_std=0,
                theta_dot_mean=0.02,
                theta_dot_std=0,
            ),
        )
    )
    environments = []
    make = gym.make

    def make_env(*args, **kwargs):
        env = make(*args, **kwargs)
        environments.append(env)
        return env

    def sample(config, rng):
        assert rng is environments[-1].unwrapped.np_random
        return sample_initial_state(config, rng)

    monkeypatch.setattr(generator.gym, "make", make_env)
    monkeypatch.setattr(generator, "sample_initial_state", sample)
    perturbation = ObservationBias(NormalOnset(0), bias=0.02) if biased else None
    store = generate_dataset(
        checkpoint=checkpoint,
        output=tmp_path / "raw",
        episodes=2,
        seed=123,
        max_episode_steps=4,
        initial_state=config,
        perturbation=perturbation,
    )
    report = GenerationReport.from_file(store.root / "generation_report.json")
    assert report.initial_state == config
    for i, tid in enumerate(store.manifest().trajectory_id):
        trajectory = store.get(tid)
        env = make("CartPole-v1", max_episode_steps=4)
        temporary_obs, _ = env.reset(seed=123 + i)
        expected = sample_initial_state(config, env.unwrapped.np_random)
        assert not np.array_equal(temporary_obs, expected.astype(np.float32))
        np.testing.assert_array_equal(trajectory.true_observations[0], expected.astype(np.float32))
        np.testing.assert_array_equal(trajectory.metadata["initial_state"], expected)
        assert trajectory.metadata["initial_state_config"] == config.model_dump(mode="json")
        env.unwrapped.state = expected.copy()
        next_obs, _, _, _, _ = env.step(int(trajectory.executed_actions[0]))
        np.testing.assert_array_equal(trajectory.true_observations[1], next_obs)
        env.close()
        assert trajectory.length == 4 and trajectory.truncated[-1]
        assert trajectory.true_observations.shape == (5, 4)
        if biased:
            assert trajectory.agent_observations[0, 2] == pytest.approx(expected[2] + 0.02)
        else:
            np.testing.assert_array_equal(
                trajectory.agent_observations, trajectory.true_observations
            )


def test_generation_seed_reproducibility(tmp_path, checkpoint):
    starts = []
    for run, seed in enumerate([123, 123, 124]):
        store = generate_dataset(
            checkpoint=checkpoint,
            output=tmp_path / str(run),
            episodes=2,
            seed=seed,
            max_episode_steps=2,
        )
        starts.append(
            [store.get(tid).true_observations[0] for tid in store.manifest().trajectory_id]
        )
    np.testing.assert_array_equal(starts[0], starts[1])
    assert not np.array_equal(starts[0][0], starts[2][0])
    np.testing.assert_array_equal(starts[0][1], starts[2][0])


def test_provenance_through_prepare_fit_embed_and_analysis(tmp_path, checkpoint):
    config = InitialStateConfig(
        distribution=Distribution.NORMAL,
        parameters=NormalInitialState(
            x_std=0.01,
            x_dot_std=0.01,
            theta_std=0.01,
            theta_dot_std=0.01,
        ),
    )
    raw = generate_dataset(
        checkpoint=checkpoint,
        output=tmp_path / "raw",
        episodes=2,
        seed=17,
        max_episode_steps=6,
        initial_state=config,
    )
    prepared = prepare_dataset(
        raw.root,
        tmp_path / "prepared",
        episode_length=3,
        min_episode_length=2,
        success_threshold=5,
        success_buffer=1,
        episode_start=1,
    )
    ids = tuple(prepared.manifest().trajectory_id)
    originals = {tid: raw.get(tid).metadata for tid in ids}
    for tid in ids:
        assert prepared.get(tid).metadata["initial_state_config"] == config.model_dump(mode="json")
        assert prepared.get(tid).metadata["initial_state"] == originals[tid]["initial_state"]
        np.testing.assert_array_equal(
            prepared.get(tid).true_observations[0], raw.get(tid).true_observations[1]
        )
    fitted = fit_dataset(
        FitConfig(dataset=prepared.root, trajectory_ids=ids, idk=IDKConfig(t=3, psi=2))
    )
    fitted.save(tmp_path / "fit")
    fitted = FitArtifact.load(tmp_path / "fit")
    embedded = embed_dataset(
        fitted,
        EmbedConfig(
            dataset=prepared.root,
            trajectory_ids=ids,
            fit_artifact=tmp_path / "fit",
            fit_id=fitted.fit_id,
            fit=fitted.config,
            unit=WindowConfig(mode="window", window_length=2),
        ),
    )
    embedded.save(tmp_path / "embedded")
    loaded = EmbeddingArtifact.load(tmp_path / "embedded")
    for unit in fitted.units + loaded.embeddings.units:
        assert unit.metadata["initial_state_config"] == config.model_dump(mode="json")
        assert unit.metadata["initial_state"] == originals[unit.trajectory_id]["initial_state"]
    execute_analysis(
        AnalysisConfig(
            command="support", embeddings=tmp_path / "embedded", output=tmp_path / "analysis"
        )
    )
    import pandas as pd

    table = pd.read_parquet(tmp_path / "analysis/support.parquet")
    assert table.initial_state_config.map(json.loads).tolist() == [
        config.model_dump(mode="json")
    ] * len(table)


def test_legacy_generation_report_without_initial_state(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": "old.pt",
                "episodes": 0,
                "seed": 1,
                "perturbation_type": "none",
                "generated": [],
            }
        )
    )
    assert GenerationReport.from_file(path).initial_state is None
