import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from click.testing import CliRunner

from cartpole_idk.cli import train as train_cli
from cartpole_idk.initial_state import reset_cartpole, sample_initial_state
from cartpole_idk.model import (
    Distribution,
    InitialStateConfig,
    NormalInitialState,
    UniformInitialState,
)
from cartpole_idk.training import TrainingConfig, evaluation, load_checkpoint, train, trainer


@pytest.mark.parametrize("distribution", list(Distribution))
def test_training_cli_initial_state_parameters(monkeypatch, distribution):
    mocked = Mock()
    monkeypatch.setattr(train_cli, "train", mocked)
    args = ["--run-dir", "run", "--initial-distribution", distribution.value]
    expected = {}
    for i, variable in enumerate(("x", "x_dot", "theta", "theta_dot")):
        values = (
            {"min": -0.1 * (i + 1), "max": 0.2 * (i + 1)}
            if distribution is Distribution.UNIFORM
            else {"mean": i * 0.01, "std": i * 0.02}
        )
        for suffix, value in values.items():
            args += [f"--initial-{variable.replace('_', '-')}-{suffix}", str(value)]
            expected[f"{variable}_{suffix}"] = value
    result = CliRunner().invoke(train_cli.main, args)
    assert result.exit_code == 0, result.output
    config = mocked.call_args.args[1].initial_state
    assert config.distribution is distribution
    assert config.parameters.model_dump() == expected


def test_training_cli_default_and_inactive_parameters(monkeypatch):
    mocked = Mock()
    monkeypatch.setattr(train_cli, "train", mocked)
    result = CliRunner().invoke(train_cli.main, ["--run-dir", "run"])
    assert result.exit_code == 0, result.output
    assert mocked.call_args.args[1].initial_state == InitialStateConfig()
    result = CliRunner().invoke(
        train_cli.main,
        ["--run-dir", "run", "--initial-distribution", "normal", "--initial-x-min", "10"],
    )
    assert result.exit_code == 0, result.output
    assert mocked.call_args.args[1].initial_state.parameters == NormalInitialState()


@pytest.mark.parametrize(
    "args",
    [
        ["--initial-distribution", "unknown"],
        ["--initial-x-min", "1", "--initial-x-max", "1"],
        ["--initial-distribution", "normal", "--initial-theta-std", "-0.1"],
        ["--initial-x-min", "nan"],
    ],
)
def test_training_cli_rejects_invalid_initial_state(monkeypatch, args):
    mocked = Mock()
    monkeypatch.setattr(train_cli, "train", mocked)
    result = CliRunner().invoke(train_cli.main, ["--run-dir", "run", *args])
    assert result.exit_code == 2, result.output
    mocked.assert_not_called()


@pytest.fixture(params=list(Distribution))
def initial_state(request):
    if request.param is Distribution.UNIFORM:
        return InitialStateConfig(parameters=UniformInitialState(x_min=0.5, x_max=0.6))
    return InitialStateConfig(
        distribution=Distribution.NORMAL,
        parameters=NormalInitialState(
            x_mean=0.5, x_std=0, x_dot_std=0, theta_std=0, theta_dot_std=0
        ),
    )


def test_reset_matches_seeded_sample_and_preserves_time_limit(initial_state):
    env = gym.make("CartPole-v1", max_episode_steps=2)
    expected_env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        for seed in (42, None, None, 42):
            expected_env.reset(seed=seed)
            expected = sample_initial_state(initial_state, expected_env.unwrapped.np_random)
            expected_env.unwrapped.state = expected.copy()
            obs = reset_cartpole(env, initial_state, seed=seed)
            np.testing.assert_array_equal(obs, expected.astype(np.float32))
            np.testing.assert_array_equal(env.unwrapped.state, expected)
            for action in (0, 1):
                actual_step = env.step(action)
                expected_step = expected_env.step(action)
                np.testing.assert_array_equal(actual_step[0], expected_step[0])
                assert actual_step[1:] == expected_step[1:]
            assert actual_step[3]  # TimeLimit still ends the episode after two steps.
    finally:
        env.close()
        expected_env.close()


def test_training_resets_replay_evaluation_and_checkpoint_metadata(
    tmp_path, monkeypatch, initial_state
):
    starts = []
    evaluation_starts = []
    transitions = []
    original_add = trainer.ReplayBuffer.add

    def training_reset(env, config, *, seed=None):
        assert config == initial_state
        obs = reset_cartpole(env, config, seed=seed)
        starts.append(obs.copy())
        return obs

    def evaluation_reset(env, config, *, seed=None):
        assert config == initial_state
        obs = reset_cartpole(env, config, seed=seed)
        evaluation_starts.append((seed, obs.copy()))
        return obs

    def add_transition(buffer, transition):
        transitions.append(transition)
        return original_add(buffer, transition)

    monkeypatch.setattr(trainer, "reset_cartpole", training_reset)
    monkeypatch.setattr(evaluation, "reset_cartpole", evaluation_reset)
    monkeypatch.setattr(trainer.ReplayBuffer, "add", add_transition)
    config = TrainingConfig(
        total_steps=4,
        max_episode_steps=2,
        hidden_sizes=(8,),
        batch_size=2,
        learning_starts=2,
        eval_every=4,
        eval_episodes=2,
        checkpoint_every=4,
        initial_state=initial_state,
    )
    train(tmp_path / "first", config)
    assert len(starts) == 3  # Initial reset, then both episode boundaries.
    assert len(transitions) == 4
    for i in range(2):
        np.testing.assert_array_equal(transitions[2 * i].obs, starts[i])
        assert transitions[2 * i + 1].done
    assert [seed for seed, _ in evaluation_starts] == [100046, 100047, 200046, 200047]
    saved_config = json.loads((tmp_path / "first/config.json").read_text())
    assert saved_config["initial_state"] == initial_state.model_dump(mode="json")
    _, checkpoint = load_checkpoint(tmp_path / "first/checkpoints/step_000000004.pt")
    assert checkpoint["metadata"]["training_config"] == config.to_dict()
    assert (
        checkpoint["metadata"]["training_config"]["initial_state"] == saved_config["initial_state"]
    )

    train(tmp_path / "repeat", config)
    np.testing.assert_array_equal(starts[:3], starts[3:])
    np.testing.assert_array_equal(
        [obs for _, obs in evaluation_starts[:4]], [obs for _, obs in evaluation_starts[4:]]
    )


def test_evaluation_policy_receives_sampled_initial_observations(initial_state):
    agent = Mock()
    agent.act.return_value = 0
    result = evaluation.evaluate_agent(
        agent, episodes=2, max_episode_steps=1, seed=123, initial_state=initial_state
    )
    env = gym.make("CartPole-v1")
    try:
        for i, call in enumerate(agent.act.call_args_list):
            expected = reset_cartpole(env, initial_state, seed=123 + i)
            np.testing.assert_array_equal(call.args[0], expected)
            assert call.kwargs == {"epsilon": 0.0}
    finally:
        env.close()
    assert agent.act.call_count == 2
    assert result["mean_episode_length"] == 1
