from __future__ import annotations

import logging

import click

from cartpole_idk.cli.options import VariadicOptionsCommand
from cartpole_idk.logging import configure_logging
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
) -> None:
    configure_logging()
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
    )
    train(run_dir, cfg, device=device)
    logger.info("Training command complete: %s", run_dir)
