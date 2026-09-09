from __future__ import annotations

import argparse
import logging

from cartpole_idk.generation import (
    ActionDelay,
    ActionFlip,
    NormalOnset,
    ObservationBias,
    Perturbation,
    generate_dataset,
)

from ._logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument(
        "--perturbation",
        choices=["none", "action-delay", "action-flip", "observation-bias"],
        default="none",
    )
    p.add_argument("--onset-mean", type=float, default=50)
    p.add_argument("--onset-std", type=float, default=5)
    p.add_argument("--delay-steps", type=int, default=3)
    p.add_argument("--flip-probability", type=float, default=0.15)
    p.add_argument(
        "--feature",
        choices=["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
        default="pole_angle",
    )
    p.add_argument("--bias", type=float, default=0.04)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--ramp-steps", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    configure_logging()
    logger.info("Generating %d episodes from %s into %s", a.episodes, a.checkpoint, a.output)
    onset = NormalOnset(a.onset_mean, a.onset_std)
    perturbation: Perturbation
    if a.perturbation == "action-delay":
        perturbation = ActionDelay(onset, a.delay_steps)
    elif a.perturbation == "action-flip":
        perturbation = ActionFlip(onset, a.flip_probability)
    elif a.perturbation == "observation-bias":
        perturbation = ObservationBias(onset, a.feature, a.bias, a.noise_std, a.ramp_steps)
    else:
        perturbation = Perturbation()
    generate_dataset(
        checkpoint=a.checkpoint,
        output=a.output,
        episodes=a.episodes,
        seed=a.seed,
        max_episode_steps=a.max_episode_steps,
        perturbation=perturbation,
        device=a.device,
    )
    logger.info("Generation command complete: %s", a.output)
