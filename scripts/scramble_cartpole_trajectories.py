#!/usr/bin/env python3
"""Create temporally scrambled copies of prepared CartPole traj_*.npz files.

A single permutation reorders the step-associated observations, actions, and
rewards in each trajectory. If observations include the extra final state
(N_steps + 1 rows), that final state is left in place. Termination/truncation
flags remain at their original timestep so that the synthetic record doesn't
advertise an early terminal event.

This preserves the exact multiset of states but deliberately breaks chronological
and physical transition relationships. The outputs are synthetic and must not be
used as valid environment rollouts.

python scripts/scramble_cartpole_trajectories.py \
    datasets/prepared/200000-iter_nominal_100_episodes_2 \
    datasets/prepared/200000-iter_nominal_100_episodes_2_scrambled \
    --seed 42
"""

import argparse
import hashlib
import json
import shutil
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("scramble_cartpole")
OBSERVATION_KEYS = ("true_observations", "agent_observations")
STEP_KEYS = ("commanded_actions", "executed_actions", "rewards")
EVENT_KEYS = ("terminated", "truncated")


def scramble_file(src: Path, dst: Path, *, seed: int, relative_path: Path) -> None:
    """Write a scrambled copy of one file without modifying the input."""
    with np.load(src, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}

    for required in ("trajectory_id", "true_observations", "metadata_json"):
        if required not in data:
            raise ValueError(f"{src}: missing required key {required!r}")

    observations = data["true_observations"]
    if observations.ndim != 2 or observations.shape[1] != 4:
        raise ValueError(
            f"{src}: expected true_observations shape (N, 4), got {observations.shape}"
        )

    # A prepared segment can have 100 observations and 100 actions, or
    # 101 observations (initial state + 100 successor states) and 100 actions.
    step_source = next((key for key in (*STEP_KEYS, *EVENT_KEYS) if key in data), None)
    if step_source is None:
        raise ValueError(f"{src}: cannot infer step count; no action/reward/flag arrays")
    n_steps = len(data[step_source])
    n_observations = len(observations)
    if n_steps < 2 or n_observations not in (n_steps, n_steps + 1):
        raise ValueError(
            f"{src}: expected >=2 steps and N or N+1 observations; "
            f"got {n_steps} {step_source} and {n_observations} observations"
        )

    for key in (*STEP_KEYS, *EVENT_KEYS):
        if key in data and (data[key].ndim < 1 or len(data[key]) != n_steps):
            raise ValueError(
                f"{src}: {key} must have {n_steps} rows, got {data[key].shape}"
            )
    for key in OBSERVATION_KEYS:
        if key in data and (data[key].ndim != 2 or len(data[key]) != n_observations):
            raise ValueError(
                f"{src}: {key} must have {n_observations} rows, got {data[key].shape}"
            )

    original_id = str(data["trajectory_id"].item())
    metadata = json.loads(str(data["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{src}: metadata_json must contain a JSON object")
    if "scramble" in metadata:
        raise ValueError(f"{src}: trajectory is already marked scrambled")

    # A reproducible but independent permutation for each source file.
    digest = hashlib.sha256(
        f"{seed}:{relative_path.as_posix()}".encode("utf-8")
    ).digest()
    file_seed = int.from_bytes(digest[:4], "little")
    permutation = np.random.default_rng(file_seed).permutation(n_steps)
    if np.array_equal(permutation, np.arange(n_steps)):
        permutation = np.roll(permutation, 1)

    for key in OBSERVATION_KEYS:
        if key in data:
            # Preserve a trailing successor observation when one exists.
            data[key] = np.concatenate(
                (data[key][:n_steps][permutation], data[key][n_steps:]),
                axis=0,
            )
    for key in STEP_KEYS:
        if key in data:
            data[key] = data[key][permutation]
    # Keep terminated/truncated at their original timestep (usually the last
    # step). They and the stored return/success describe the ORIGINAL rollout.

    metadata["scramble"] = {
        "method": "joint_step_permutation_with_final_observation_preserved",
        "seed": seed,
        "file_seed": file_seed,
        "step_permutation": permutation.tolist(),
        "final_observation_preserved": n_observations == n_steps + 1,
        "termination_flags_preserved": True,
        "original_trajectory_id": original_id,
        "source_relative_path": relative_path.as_posix(),
    }
    metadata["is_synthetic"] = True
    data["trajectory_id"] = np.asarray(f"{original_id}")
    data["metadata_json"] = np.asarray(json.dumps(metadata))

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **data)
    LOG.info("Scrambled %s -> %s (%d steps, %d observations)",
             src, dst, n_steps, n_observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of traj_*.npz")
    parser.add_argument("output_dir", type=Path, help="Separate output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = args.input_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error("input_dir must be a directory")
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("output_dir must be separate from, and not inside, input_dir")

    files = sorted(input_dir.rglob("traj_*.npz"))
    LOG.info("Found %d trajectory files", len(files))
    if not files:
        parser.error("No traj_*.npz files found")

    for src in files:
        relative = src.relative_to(input_dir)
        dst = output_dir / relative.parent / f"{src.stem}.npz"
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"{dst} already exists (use --overwrite)")
        scramble_file(src, dst, seed=args.seed, relative_path=relative)

    for name in ("manifest.parquet", "preparation_report.json", "generation_report.json"):
        src = input_dir / name
        if src.exists():
            dst = output_dir / name
            if dst.exists() and not args.overwrite:
                raise FileExistsError(f"{dst} already exists (use --overwrite)")
            shutil.copy2(src, dst)
            LOG.info("Copied %s -> %s", src, dst)

    LOG.info("Complete: scrambled %d trajectories", len(files))


if __name__ == "__main__":
    main()
