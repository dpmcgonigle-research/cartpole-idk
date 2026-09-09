import argparse
import logging

from cartpole_idk.idk import IDKExperimentConfig, fit_reference, trajectory_familiarity
from cartpole_idk.storage import TrajectoryStore

from ._logging import configure_logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--reference-ids", nargs="+", required=True)
    p.add_argument("--query-ids", nargs="+", required=True)
    p.add_argument(
        "--representation",
        choices=["state", "transition", "state_action", "state_action_next_state", "window"],
        default="state",
    )
    p.add_argument("--window-length", type=int, default=25)
    p.add_argument("--psi", type=int, default=32)
    p.add_argument("--t", type=int, default=200)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    configure_logging()
    logger.info("Fitting IDK reference from %d trajectories in %s", len(a.reference_ids), a.dataset)
    store = TrajectoryStore(a.dataset)
    cfg = IDKExperimentConfig(
        representation=a.representation, window_length=a.window_length, psi=a.psi, t=a.t
    )
    ref = fit_reference(store, a.reference_ids, cfg)
    logger.info("Scoring %d query trajectories", len(a.query_ids))
    scores = trajectory_familiarity(ref, [store.get(x) for x in a.query_ids], k=a.k)
    for tid, score in zip(a.query_ids, scores, strict=True):
        print(f"{tid}\t{score:.6f}")
    logger.info("IDK evaluation complete: %d scores", len(scores))
