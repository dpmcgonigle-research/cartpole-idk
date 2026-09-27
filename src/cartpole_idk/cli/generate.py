from __future__ import annotations

import logging

import click

from cartpole_idk.generation import (
    ActionDelay,
    ActionFlip,
    NormalOnset,
    ObservationBias,
    Perturbation,
    generate_dataset,
)
from cartpole_idk.logging import configure_logging
from cartpole_idk.model import (
    Distribution,
    InitialStateConfig,
    NormalInitialState,
    UniformInitialState,
)

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--checkpoint", required=True, type=str)
@click.option("--output", required=True, type=str)
@click.option("--episodes", default=100, type=int)
@click.option("--seed", default=1000, type=int)
@click.option("--max-episode-steps", default=500, type=int)
@click.option(
    "--perturbation",
    default="none",
    type=click.Choice(["none", "action-delay", "action-flip", "observation-bias"]),
)
@click.option("--onset-mean", default=50, type=float)
@click.option("--onset-std", default=5, type=float)
@click.option("--delay-steps", default=3, type=int)
@click.option("--flip-probability", default=0.15, type=float)
@click.option(
    "--feature",
    default="pole_angle",
    type=click.Choice(["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]),
)
@click.option("--bias", default=0.04, type=float)
@click.option("--noise-std", default=0.0, type=float)
@click.option("--ramp-steps", default=0, type=int)
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
    checkpoint,
    output,
    episodes,
    seed,
    max_episode_steps,
    perturbation,
    onset_mean,
    onset_std,
    delay_steps,
    flip_probability,
    feature,
    bias,
    noise_std,
    ramp_steps,
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
    """Generate trajectory data and a return-statistics report from a policy.

    \b
    Args:
        checkpoint: Trained policy checkpoint to load.
        output: Destination dataset directory.
        episodes: Number of trajectories to generate.
        seed: Base random seed for episode generation.
        max_episode_steps: Maximum transitions per trajectory.
        perturbation: Perturbation mechanism, or none for nominal rollouts.
        onset_mean: Mean perturbation start timestep (zero-based).
        onset_std: Standard deviation of normally sampled start timesteps.
        delay_steps: Action delay in transitions for action-delay.
        flip_probability: Chance of reversing an action for action-flip.
        feature: Observation component affected by observation-bias.
        bias: Additive offset for observation-bias.
        noise_std: Standard deviation of additive observation noise.
        ramp_steps: Steps to reach full bias; zero applies it immediately.
        device: PyTorch device used for policy inference.
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

        Rollouts use greedy policy actions. Perturbation-specific settings
        apply only to the selected mechanism; true and agent states are saved.
        Initial-state parameters apply only to the selected distribution.
        Sampling uses the environment RNG after reset; normal samples are not clipped.
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
    logger.info("Generating %d episodes from %s into %s", episodes, checkpoint, output)
    onset = NormalOnset(onset_mean, onset_std)
    selected_perturbation: Perturbation
    if perturbation == "action-delay":
        selected_perturbation = ActionDelay(onset, delay_steps)
    elif perturbation == "action-flip":
        selected_perturbation = ActionFlip(onset, flip_probability)
    elif perturbation == "observation-bias":
        selected_perturbation = ObservationBias(onset, feature, bias, noise_std, ramp_steps)
    else:
        selected_perturbation = Perturbation()
    generate_dataset(
        checkpoint=checkpoint,
        output=output,
        episodes=episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
        perturbation=selected_perturbation,
        device=device,
        initial_state=initial_state,
    )
    logger.info("Generation command complete: %s", output)
