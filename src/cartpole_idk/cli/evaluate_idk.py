import logging

import click

from cartpole_idk.cli.options import VariadicOptionsCommand
from cartpole_idk.idk import IDKExperimentConfig, fit_reference, trajectory_familiarity
from cartpole_idk.logging import configure_logging
from cartpole_idk.storage import TrajectoryStore

logger = logging.getLogger(__name__)


@click.command(cls=VariadicOptionsCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset")
@click.option("--reference-ids", required=True, multiple=True, type=str)
@click.option("--query-ids", required=True, multiple=True, type=str)
@click.option(
    "--representation",
    default="state",
    type=click.Choice(["state", "transition", "state_action", "state_action_next_state", "window"]),
)
@click.option("--window-length", default=25, type=int)
@click.option("--psi", default=32, type=int)
@click.option("--t", default=200, type=int)
@click.option("--k", default=5, type=int)
def main(dataset, reference_ids, query_ids, representation, window_length, psi, t, k) -> None:
    configure_logging()
    logger.info("Fitting IDK reference from %d trajectories in %s", len(reference_ids), dataset)
    store = TrajectoryStore(dataset)
    cfg = IDKExperimentConfig(
        representation=representation, window_length=window_length, psi=psi, t=t
    )
    ref = fit_reference(store, reference_ids, cfg)
    logger.info("Scoring %d query trajectories", len(query_ids))
    scores = trajectory_familiarity(ref, [store.get(x) for x in query_ids], k=k)
    for tid, score in zip(query_ids, scores, strict=True):
        print(f"{tid}\t{score:.6f}")
    logger.info("IDK evaluation complete: %d scores", len(scores))
