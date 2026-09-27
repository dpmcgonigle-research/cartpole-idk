from __future__ import annotations

import logging

import click

from cartpole_idk.cli.options import VariadicOptionsCommand
from cartpole_idk.logging import configure_logging
from cartpole_idk.model import (
    Distribution,
    InitialStateConfig,
    NormalInitialState,
    UniformInitialState,
)
from cartpole_idk.training import TrainingConfig, train

logger = logging.getLogger(__name__)


@click.command(cls=VariadicOptionsCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--run-dir", required=True, type=str)
@click.option("--total-steps", default=200000, type=int)
@click.option("--max-episode-steps", default=500, type=int)
@click.option("--hidden-sizes", default=[128, 128], multiple=True, type=int)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--eval-every", default=5000, type=int)
@click.option("--eval-episodes", default=20, type=int)
@click.option("--checkpoint-every", default=10000, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--device", default="cpu", type=str)
@click.option(
    "--initial-distribution",
    type=click.Choice([distribution.value for distribution in Distribution]),
    default=Distribution.UNIFORM.value,
    show_default=True,
    help="Independent starting-state distribution for all four state variables",
)
@click.option(
    "--initial-x-min",
    type=float,
    default=-0.05,
    show_default=True,
    help="Uniform lower bound for x",
)
@click.option(
    "--initial-x-max", type=float, default=0.05, show_default=True, help="Uniform upper bound for x"
)
@click.option(
    "--initial-x-mean", type=float, default=0.0, show_default=True, help="Normal mean for x"
)
@click.option(
    "--initial-x-std",
    type=float,
    default=0.05,
    show_default=True,
    help="Normal standard deviation (zero fixes this component) for x",
)
@click.option(
    "--initial-x-dot-min",
    type=float,
    default=-0.05,
    show_default=True,
    help="Uniform lower bound for x_dot",
)
@click.option(
    "--initial-x-dot-max",
    type=float,
    default=0.05,
    show_default=True,
    help="Uniform upper bound for x_dot",
)
@click.option(
    "--initial-x-dot-mean", type=float, default=0.0, show_default=True, help="Normal mean for x_dot"
)
@click.option(
    "--initial-x-dot-std",
    type=float,
    default=0.05,
    show_default=True,
    help="Normal standard deviation (zero fixes this component) for x_dot",
)
@click.option(
    "--initial-theta-min",
    type=float,
    default=-0.05,
    show_default=True,
    help="Uniform lower bound for theta",
)
@click.option(
    "--initial-theta-max",
    type=float,
    default=0.05,
    show_default=True,
    help="Uniform upper bound for theta",
)
@click.option(
    "--initial-theta-mean", type=float, default=0.0, show_default=True, help="Normal mean for theta"
)
@click.option(
    "--initial-theta-std",
    type=float,
    default=0.05,
    show_default=True,
    help="Normal standard deviation (zero fixes this component) for theta",
)
@click.option(
    "--initial-theta-dot-min",
    type=float,
    default=-0.05,
    show_default=True,
    help="Uniform lower bound for theta_dot",
)
@click.option(
    "--initial-theta-dot-max",
    type=float,
    default=0.05,
    show_default=True,
    help="Uniform upper bound for theta_dot",
)
@click.option(
    "--initial-theta-dot-mean",
    type=float,
    default=0.0,
    show_default=True,
    help="Normal mean for theta_dot",
)
@click.option(
    "--initial-theta-dot-std",
    type=float,
    default=0.05,
    show_default=True,
    help="Normal standard deviation (zero fixes this component) for theta_dot",
)
def main(
    run_dir,
    total_steps,
    max_episode_steps,
    hidden_sizes,
    learning_rate,
    batch_size,
    eval_every,
    eval_episodes,
    checkpoint_every,
    seed,
    device,
    initial_distribution: str,
    initial_x_min: float,
    initial_x_max: float,
    initial_x_mean: float,
    initial_x_std: float,
    initial_x_dot_min: float,
    initial_x_dot_max: float,
    initial_x_dot_mean: float,
    initial_x_dot_std: float,
    initial_theta_min: float,
    initial_theta_max: float,
    initial_theta_mean: float,
    initial_theta_std: float,
    initial_theta_dot_min: float,
    initial_theta_dot_max: float,
    initial_theta_dot_mean: float,
    initial_theta_dot_std: float,
) -> None:
    """Train a CartPole DQN policy and save checkpoints and metrics.

    \b
    Args:
        run_dir: Destination directory for the training run.
        total_steps: Total environment interactions used for training.
        max_episode_steps: Maximum transitions per episode.
        hidden_sizes: Hidden-layer widths, supplied as space-separated integers.
        learning_rate: Optimizer step size.
        batch_size: Replay-buffer samples per training update.
        eval_every: Environment steps between policy evaluations.
        eval_episodes: Greedy episodes per evaluation.
        checkpoint_every: Environment steps between saved checkpoints.
        seed: Random seed for training.
        device: PyTorch device, such as cpu or cuda.
        initial_distribution: Uniform bounds or untruncated Gaussian sampling.
        initial_x_min: Uniform lower bound for x.
        initial_x_max: Uniform upper bound for x.
        initial_x_mean: Normal mean for x.
        initial_x_std: Normal standard deviation (zero fixes this component) for x.
        initial_x_dot_min: Uniform lower bound for x_dot.
        initial_x_dot_max: Uniform upper bound for x_dot.
        initial_x_dot_mean: Normal mean for x_dot.
        initial_x_dot_std: Normal standard deviation (zero fixes this component) for x_dot.
        initial_theta_min: Uniform lower bound for theta.
        initial_theta_max: Uniform upper bound for theta.
        initial_theta_mean: Normal mean for theta.
        initial_theta_std: Normal standard deviation (zero fixes this component) for theta.
        initial_theta_dot_min: Uniform lower bound for theta_dot.
        initial_theta_dot_max: Uniform upper bound for theta_dot.
        initial_theta_dot_mean: Normal mean for theta_dot.
        initial_theta_dot_std: Normal standard deviation (zero fixes this component) for theta_dot.

    Notes:

        Evaluation uses greedy actions without learning. Evaluation and
        checkpoint intervals count environment steps, not episodes.
        Training and evaluation use the same initial-state distribution.
        Angles are radians; angular velocities are radians per second.
        Only the selected distribution parameters apply; samples are not clipped.
    """
    configure_logging()
    try:
        distribution = Distribution(initial_distribution)
        parameters: UniformInitialState | NormalInitialState
        if distribution is Distribution.UNIFORM:
            parameters = UniformInitialState(
                x_min=initial_x_min,
                x_max=initial_x_max,
                x_dot_min=initial_x_dot_min,
                x_dot_max=initial_x_dot_max,
                theta_min=initial_theta_min,
                theta_max=initial_theta_max,
                theta_dot_min=initial_theta_dot_min,
                theta_dot_max=initial_theta_dot_max,
            )
        else:
            parameters = NormalInitialState(
                x_mean=initial_x_mean,
                x_std=initial_x_std,
                x_dot_mean=initial_x_dot_mean,
                x_dot_std=initial_x_dot_std,
                theta_mean=initial_theta_mean,
                theta_std=initial_theta_std,
                theta_dot_mean=initial_theta_dot_mean,
                theta_dot_std=initial_theta_dot_std,
            )
        initial_state = InitialStateConfig(distribution=distribution, parameters=parameters)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    logger.info("Training requested: run_dir=%s, device=%s", run_dir, device)
    cfg = TrainingConfig(
        total_steps=total_steps,
        max_episode_steps=max_episode_steps,
        hidden_sizes=tuple(hidden_sizes),
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        checkpoint_every=checkpoint_every,
        seed=seed,
        initial_state=initial_state,
    )
    train(run_dir, cfg, device=device)
    logger.info("Training command complete: %s", run_dir)
