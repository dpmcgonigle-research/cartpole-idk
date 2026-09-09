import argparse
import logging

from cartpole_idk.storage import TrajectoryStore

from ._logging import configure_logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--checkpoint-id")
    p.add_argument("--perturbation")
    p.add_argument("--min-return", type=float)
    p.add_argument("--max-return", type=float)
    p.add_argument("--min-length", type=int)
    p.add_argument("--max-length", type=int)
    a = p.parse_args()
    configure_logging()
    logger.info("Querying trajectories in %s", a.dataset)
    df = TrajectoryStore(a.dataset).query(
        checkpoint_id=a.checkpoint_id,
        perturbation_type=a.perturbation,
        min_return=a.min_return,
        max_return=a.max_return,
        min_length=a.min_length,
        max_length=a.max_length,
    )
    print(df.to_string(index=False))
    logger.info("Query complete: %d matching trajectories", len(df))
