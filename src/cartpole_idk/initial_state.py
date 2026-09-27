"""Independent CartPole starting-state sampling using the environment's RNG."""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from cartpole_idk.model import (
    Distribution,
    InitialStateConfig,
    NormalInitialState,
    UniformInitialState,
)


def sample_initial_state(config: InitialStateConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample [x, x_dot, theta, theta_dot] without clipping or truncation.

    Args:
        config: Validated distribution and its four independent parameter pairs.
        rng: Seeded env.unwrapped.np_random, used after the environment reset.
    """
    parameters = config.parameters
    if config.distribution is Distribution.UNIFORM:
        assert isinstance(parameters, UniformInitialState)
        return rng.uniform(
            low=[
                parameters.x_min,
                parameters.x_dot_min,
                parameters.theta_min,
                parameters.theta_dot_min,
            ],
            high=[
                parameters.x_max,
                parameters.x_dot_max,
                parameters.theta_max,
                parameters.theta_dot_max,
            ],
        )
    assert config.distribution is Distribution.NORMAL
    assert isinstance(parameters, NormalInitialState)
    return rng.normal(
        loc=[
            parameters.x_mean,
            parameters.x_dot_mean,
            parameters.theta_mean,
            parameters.theta_dot_mean,
        ],
        scale=[
            parameters.x_std,
            parameters.x_dot_std,
            parameters.theta_std,
            parameters.theta_dot_std,
        ],
    )


def reset_cartpole(
    env: gym.Env, config: InitialStateConfig, *, seed: int | None = None
) -> np.ndarray:
    """Reset episode bookkeeping and return the sampled starting observation.

    Args:
        env: Wrapped CartPole environment; wrappers remain active.
        config: Independent starting-state distribution and parameters.
        seed: Environment seed; None continues its current random stream.
    """
    env.reset(seed=seed)
    cartpole = cast(CartPoleEnv, env.unwrapped)
    state = sample_initial_state(config, cartpole.np_random)
    cartpole.state = state.copy()
    return np.asarray(state, dtype=np.float32)
