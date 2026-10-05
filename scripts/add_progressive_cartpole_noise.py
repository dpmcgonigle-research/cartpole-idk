#!/usr/bin/env python3
"""Create synthetic CartPole trajectories with progressively increasing Gaussian noise.

The script recursively finds ``traj_*.npz`` files under INPUT_DIR, copies each
archive to the same relative path under OUTPUT_DIR, and adds zero-mean Gaussian
noise to one CartPole observation component.  The noise standard deviation grows
monotonically along each trajectory according to

    sigma(p) = start_std + (end_std - start_std) * p**power

where p runs from 0 to 1 after ``--onset``.  Before onset, sigma is zero.

Only observation arrays are modified.  Actions, rewards, termination flags,
trajectory IDs, and all other arrays are preserved.  Metadata is augmented with
a compact ``progressive_noise`` object so the transformation is reproducible.

Examples
--------
Add increasing noise to pole angle in both true and agent observations:

    python scripts/add_progressive_cartpole_noise.py \
        datasets/prepared/200000-iter_nominal_100_episodes_2 \
        datasets/prepared/200000-iter_nominal_100_episodes_2_theta_noise \
        --dimension theta \
        --start-std 0.0 \
        --end-std 0.25 \
        --seed 42

Add increasing noise to cart velocity only in agent observations:

    python scripts/add_progressive_cartpole_noise.py \
        INPUT_DIR OUTPUT_DIR \
        --dimension x_dot \
        --start-std 0.0 \
        --end-std 0.5 \
        --target agent \
        --onset 20 \
        --power 1.0 \
        --seed 42

Notes
-----
* ``start_std`` and ``end_std`` are in the RAW units of the selected CartPole
  component.  Your saved scaler will subsequently convert these to standardized
  units during embedding.
* The standard-deviation envelope is monotonic.  Individual Gaussian samples are
  random, so their absolute magnitudes are not themselves monotonic.
* With ``--target both`` the exact same noise realization is applied to
  ``true_observations`` and ``agent_observations`` so those arrays remain aligned.
* Output trajectory filenames and trajectory IDs are unchanged, which makes
  clean/noisy pairing straightforward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np

LOG = logging.getLogger("progressive_cartpole_noise")

DIMENSIONS = {
    "x": 0,
    "x_dot": 1,
    "xdot": 1,
    "theta": 2,
    "theta_dot": 3,
    "thetadot": 3,
}
CANONICAL_NAMES = ("x", "x_dot", "theta", "theta_dot")
TARGETS = {
    "true": ("true_observations",),
    "agent": ("agent_observations",),
    "both": ("true_observations", "agent_observations"),
}


def parse_dimension(value: str) -> int:
    """Parse dimension name or integer index."""
    normalized = value.strip().lower()
    if normalized in DIMENSIONS:
        return DIMENSIONS[normalized]
    try:
        index = int(normalized)
    except ValueError as exc:
        choices = ", ".join(CANONICAL_NAMES)
        raise argparse.ArgumentTypeError(
            f"dimension must be 0..3 or one of: {choices}"
        ) from exc
    if index not in range(4):
        raise argparse.ArgumentTypeError("dimension index must be in 0..3")
    return index


def progressive_sigma(
    n: int,
    *,
    start_std: float,
    end_std: float,
    onset: int,
    power: float,
) -> np.ndarray:
    """Return one monotonically nondecreasing sigma value per observation."""
    if n < 1:
        raise ValueError("observation array must contain at least one row")
    if not 0 <= onset < n:
        raise ValueError(f"onset must satisfy 0 <= onset < {n}; got {onset}")

    sigma = np.zeros(n, dtype=np.float64)
    remaining = n - onset
    if remaining == 1:
        sigma[onset] = end_std
        return sigma

    phase = np.linspace(0.0, 1.0, remaining, dtype=np.float64)
    sigma[onset:] = start_std + (end_std - start_std) * np.power(phase, power)
    return sigma


def noise_file(
    src: Path,
    dst: Path,
    *,
    seed: int,
    relative_path: Path,
    dimension: int,
    start_std: float,
    end_std: float,
    onset: int,
    power: float,
    target: str,
) -> None:
    """Write one progressively noised trajectory copy."""
    with np.load(src, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}

    for required in ("trajectory_id", "metadata_json"):
        if required not in data:
            raise ValueError(f"{src}: missing required key {required!r}")

    target_keys = TARGETS[target]
    missing = [key for key in target_keys if key not in data]
    if missing:
        raise ValueError(f"{src}: missing requested observation arrays: {missing}")

    lengths = set()
    for key in target_keys:
        observations = data[key]
        if observations.ndim != 2 or observations.shape[1] != 4:
            raise ValueError(
                f"{src}: expected {key} shape (N, 4), got {observations.shape}"
            )
        lengths.add(observations.shape[0])

    if len(lengths) != 1:
        raise ValueError(
            f"{src}: target observation arrays have different lengths: {sorted(lengths)}"
        )
    n_observations = lengths.pop()

    metadata = json.loads(str(data["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{src}: metadata_json must contain a JSON object")
    if "progressive_noise" in metadata:
        raise ValueError(f"{src}: trajectory is already marked with progressive noise")

    # 32-bit deterministic per-file seed: ample for this experiment and safely
    # representable in downstream JSON/Parquet tooling.
    digest = hashlib.sha256(
        f"{seed}:{relative_path.as_posix()}".encode("utf-8")
    ).digest()
    file_seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(file_seed)

    sigma = progressive_sigma(
        n_observations,
        start_std=start_std,
        end_std=end_std,
        onset=onset,
        power=power,
    )
    noise = rng.normal(loc=0.0, scale=sigma, size=n_observations)

    # Reuse the exact same realization for both arrays when target == "both".
    for key in target_keys:
        updated = data[key].astype(np.float64, copy=True)
        updated[:, dimension] += noise
        data[key] = updated

    metadata["progressive_noise"] = {
        "distribution": "normal",
        "dimension": dimension,
        "dimension_name": CANONICAL_NAMES[dimension],
        "start_std": float(start_std),
        "end_std": float(end_std),
        "onset": int(onset),
        "power": float(power),
        "target": target,
        "target_arrays": list(target_keys),
        "seed": int(seed),
        "file_seed": int(file_seed),
        "source_relative_path": relative_path.as_posix(),
        "observation_count": int(n_observations),
    }
    metadata["is_synthetic"] = True
    data["metadata_json"] = np.asarray(json.dumps(metadata))

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **data)
    LOG.info(
        "Noised %s -> %s (%s, std %.6g -> %.6g, onset=%d)",
        src,
        dst,
        CANONICAL_NAMES[dimension],
        start_std,
        end_std,
        onset,
    )


def copy_dataset_metadata(input_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    """Copy the prepared-dataset sidecars used elsewhere in cartpole-idk."""
    for name in ("manifest.parquet", "preparation_report.json"):
        src = input_dir / name
        if not src.exists():
            continue
        dst = output_dir / name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} already exists (use --overwrite)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        LOG.info("Copied %s -> %s", src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Generated or prepared dataset directory")
    parser.add_argument("output_dir", type=Path, help="Separate output directory")
    parser.add_argument(
        "--dimension",
        required=True,
        type=parse_dimension,
        metavar="{x,x_dot,theta,theta_dot,0,1,2,3}",
        help="CartPole state component to corrupt",
    )
    parser.add_argument(
        "--start-std",
        type=float,
        default=0.0,
        help="Gaussian std at the first noised observation (raw feature units)",
    )
    parser.add_argument(
        "--end-std",
        type=float,
        required=True,
        help="Gaussian std at the final observation (raw feature units)",
    )
    parser.add_argument(
        "--onset",
        type=int,
        default=0,
        help="Observation index where the progressive-noise schedule begins",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="Schedule exponent: 1=linear, >1 slower early growth, <1 faster early growth",
    )
    parser.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default="both",
        help="Which observation arrays to modify (default: both)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.start_std < 0 or args.end_std < 0:
        parser.error("--start-std and --end-std must be >= 0")
    if args.end_std < args.start_std:
        parser.error("--end-std must be >= --start-std for a monotonic increase")
    if args.power <= 0:
        parser.error("--power must be > 0")
    if args.onset < 0:
        parser.error("--onset must be >= 0")

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
        dst = output_dir / relative  # preserve filename and relative structure
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"{dst} already exists (use --overwrite)")
        noise_file(
            src,
            dst,
            seed=args.seed,
            relative_path=relative,
            dimension=args.dimension,
            start_std=args.start_std,
            end_std=args.end_std,
            onset=args.onset,
            power=args.power,
            target=args.target,
        )

    copy_dataset_metadata(input_dir, output_dir, overwrite=args.overwrite)

    sidecar = {
        "transformation": "progressive_gaussian_noise",
        "dimension": args.dimension,
        "dimension_name": CANONICAL_NAMES[args.dimension],
        "start_std": args.start_std,
        "end_std": args.end_std,
        "onset": args.onset,
        "power": args.power,
        "target": args.target,
        "seed": args.seed,
        "input_dir": str(input_dir),
        "trajectory_count": len(files),
    }
    sidecar_path = output_dir / "noise_injection.json"
    if sidecar_path.exists() and not args.overwrite:
        raise FileExistsError(f"{sidecar_path} already exists (use --overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    LOG.info("Wrote %s", sidecar_path)
    LOG.info("Complete: generated %d progressively noised trajectories", len(files))


if __name__ == "__main__":
    main()
