#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)

STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")
NUM_STATE_DIMS = len(STATE_NAMES)


def find_state_bounds(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recursively find CartPole trajectory files and compute state bounds."""

    traj_files = sorted(directory.rglob("traj_*.npz"))

    LOGGER.info("Discovered %d trajectory files under %s", len(traj_files), directory)

    if not traj_files:
        raise FileNotFoundError(
            f"No traj_*.npz files found under: {directory}"
        )

    global_min = np.full(NUM_STATE_DIMS, np.inf, dtype=np.float64)
    global_max = np.full(NUM_STATE_DIMS, -np.inf, dtype=np.float64)

    for index, filepath in enumerate(traj_files, start=1):
        try:
            with np.load(filepath) as data:
                if "true_observations" not in data:
                    raise KeyError(
                        f"{filepath} does not contain 'true_observations'"
                    )

                observations = np.asarray(
                    data["true_observations"],
                    dtype=np.float64,
                )

            if observations.ndim != 2:
                raise ValueError(
                    f"{filepath}: expected true_observations to be 2-D, "
                    f"got shape {observations.shape}"
                )

            if observations.shape[1] != NUM_STATE_DIMS:
                raise ValueError(
                    f"{filepath}: expected {NUM_STATE_DIMS} state dimensions "
                    f"[x, x_dot, theta, theta_dot], "
                    f"got shape {observations.shape}"
                )

            if observations.shape[0] == 0:
                raise ValueError(
                    f"{filepath}: true_observations contains no observations"
                )

            file_min = np.min(observations, axis=0)
            file_max = np.max(observations, axis=0)

            global_min = np.minimum(global_min, file_min)
            global_max = np.maximum(global_max, file_max)

            LOGGER.info(
                "Completed %d/%d: %s",
                index,
                len(traj_files),
                filepath,
            )

        except Exception:
            LOGGER.exception("Failed processing %s", filepath)
            raise

    return global_min, global_max


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively search for CartPole traj_*.npz files and compute "
            "min/max values from their 'true_observations' arrays."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory containing trajectory files.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    directory = args.directory.expanduser().resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    global_min, global_max = find_state_bounds(directory)

    result = ", ".join(
        f"{name}: min={min_value:.10f}, max={max_value:.10f}"
        for name, min_value, max_value in zip(
            STATE_NAMES,
            global_min,
            global_max,
        )
    )

    LOGGER.info("Final state-space bounds: %s", result)


if __name__ == "__main__":
    main()
