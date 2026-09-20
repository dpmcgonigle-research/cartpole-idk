import logging

import click

from cartpole_idk.logging import configure_logging
from cartpole_idk.storage import TrajectoryStore
from cartpole_idk.visualization import replay_trajectory

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset")
@click.option("--trajectory", required=True, type=str)
@click.option("--fps", default=50.0, type=float)
def main(dataset, trajectory, fps) -> None:
    configure_logging()
    logger.info("Loading trajectory %s from %s for replay at %g FPS", trajectory, dataset, fps)
    replay_trajectory(TrajectoryStore(dataset).get(trajectory), fps=fps)
    logger.info("Replay finished: %s", trajectory)
