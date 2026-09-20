import logging

import click

from cartpole_idk.logging import configure_logging
from cartpole_idk.storage import TrajectoryStore

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset")
@click.option("--checkpoint-id", type=str)
@click.option("--perturbation", type=str)
@click.option("--min-return", type=float)
@click.option("--max-return", type=float)
@click.option("--min-length", type=int)
@click.option("--max-length", type=int)
def main(
    dataset, checkpoint_id, perturbation, min_return, max_return, min_length, max_length
) -> None:
    configure_logging()
    logger.info("Querying trajectories in %s", dataset)
    df = TrajectoryStore(dataset).query(
        checkpoint_id=checkpoint_id,
        perturbation_type=perturbation,
        min_return=min_return,
        max_return=max_return,
        min_length=min_length,
        max_length=max_length,
    )
    print(df.to_string(index=False))
    logger.info("Query complete: %d matching trajectories", len(df))
