import argparse
import logging

from cartpole_idk.storage import TrajectoryStore
from cartpole_idk.visualization import replay_trajectory

from ._logging import configure_logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--trajectory", required=True)
    p.add_argument("--fps", type=float, default=50.0)
    a = p.parse_args()
    configure_logging()
    logger.info(
        "Loading trajectory %s from %s for replay at %g FPS", a.trajectory, a.dataset, a.fps
    )
    replay_trajectory(TrajectoryStore(a.dataset).get(a.trajectory), fps=a.fps)
    logger.info("Replay finished: %s", a.trajectory)
