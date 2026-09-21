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

    Notes:

        Rollouts use greedy policy actions. Perturbation-specific settings
        apply only to the selected mechanism; true and agent states are saved.
    """
    configure_logging()
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
    )
    logger.info("Generation command complete: %s", output)
