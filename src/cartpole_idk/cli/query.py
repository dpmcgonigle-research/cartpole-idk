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
    """Print trajectory metadata matching the supplied filters.

    \b
    Args:
        dataset: Trajectory dataset directory to query.
        checkpoint_id: Exact checkpoint identifier to match.
        perturbation: Stored perturbation type, such as action_delay.
        min_return: Inclusive lower bound on summed episode rewards.
        max_return: Inclusive upper bound on summed episode rewards.
        min_length: Inclusive minimum trajectory length in transitions.
        max_length: Inclusive maximum trajectory length in transitions.

    Notes:

        Omitted filters impose no restriction. The table goes to stdout;
        diagnostic logging goes to stderr.
    """
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
