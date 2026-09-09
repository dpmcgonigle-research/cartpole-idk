from __future__ import annotations

import argparse
import logging

from cartpole_idk.training import TrainingConfig, train

from ._logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    configure_logging()
    logger.info("Training requested: run_dir=%s, device=%s", a.run_dir, a.device)
    cfg = TrainingConfig(
        total_steps=a.total_steps,
        max_episode_steps=a.max_episode_steps,
        hidden_sizes=tuple(a.hidden_sizes),
        learning_rate=a.learning_rate,
        batch_size=a.batch_size,
        eval_every=a.eval_every,
        eval_episodes=a.eval_episodes,
        checkpoint_every=a.checkpoint_every,
        seed=a.seed,
    )
    train(a.run_dir, cfg, device=a.device)
    logger.info("Training command complete: %s", a.run_dir)
