"""Select contiguous trajectory segments for IDK fitting."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cartpole_idk.storage import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)


def prepare_dataset(
    dataset: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    episode_length: int = 100,
    min_episode_length: int = 25,
    success_threshold: int = 300,
    success_buffer: int = 50,
    episode_start: int | None = None,
    seed: int = 1000,
) -> TrajectoryStore:
    """Save one segment per eligible source trajectory, with source provenance.

    Lengths count transitions; observations include the final next state.
    An explicit start overrides failure-tail selection as well as random starts.

    Args:
        dataset: One or more generated dataset directories to combine.
        output: New or empty destination dataset directory.
        episode_length: Maximum number of transitions in a retained segment.
        min_episode_length: Minimum retained segment length; shorter segments are skipped.
        success_threshold: Source trajectory length required to classify a run as successful.
        success_buffer: Gap before the success threshold within which segments must end.
        episode_start: Fixed start, or None for uniform success starts and failure tails.
        seed: Random seed for successful-segment sampling.
    """
    if not 1 <= min_episode_length <= episode_length:
        raise ValueError("Require 1 <= min_episode_length <= episode_length")
    if success_threshold <= 0 or not 0 <= success_buffer < success_threshold:
        raise ValueError(
            "Require success_threshold > 0 and 0 <= success_buffer < success_threshold"
        )
    latest_start = success_threshold - success_buffer - episode_length
    if latest_start < 0:
        raise ValueError("episode_length must fit within success_threshold - success_buffer")
    if episode_start is not None and not 0 <= episode_start <= latest_start:
        raise ValueError("episode_start must allow a full segment inside the success boundary")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    datasets = [dataset] if isinstance(dataset, (str, Path)) else list(dataset)
    if not datasets:
        raise ValueError("At least one source dataset is required")

    logger.info(f"Preparing dataset from {datasets}")
    logger.info(f"episode_length: {episode_length}")
    logger.info(f"min_episode_length: {min_episode_length}")
    logger.info(f"success_threshold: {success_threshold}")
    logger.info(f"success_buffer: {success_buffer}")

    sources = [TrajectoryStore(path) for path in datasets]
    source_paths = [str(source.root.resolve()) for source in sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("Source datasets must be distinct")
    destination = Path(output)
    for source in sources:
        if not source.manifest_path.is_file():
            raise ValueError(f"Dataset manifest not found: {source.manifest_path}")
        if destination.resolve() == source.root.resolve():
            raise ValueError("Output must be different from every source dataset")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    manifests = [source.manifest() for source in sources]
    rng = np.random.default_rng(seed)
    store = TrajectoryStore(destination)
    store.initialize()
    saved = 0
    skipped = 0
    for source_index, (source, rows) in enumerate(zip(sources, manifests, strict=True)):
        for tid in rows.get("trajectory_id", []):
            trajectory = source.get(tid)
            success = trajectory.length >= success_threshold
            if episode_start is not None:
                start = episode_start
            elif success:
                start = int(rng.integers(0, latest_start + 1))
            else:
                start = max(0, trajectory.length - episode_length)
            stop = min(start + episode_length, trajectory.length)
            if stop - start < min_episode_length:
                skipped += 1
                continue
            metadata = {
                **trajectory.metadata,
                "source_dataset": str(source.root.resolve()),
                "source_trajectory_id": trajectory.trajectory_id,
                "source_length": trajectory.length,
                "source_return": trajectory.episode_return,
                "success": success,
                "segment_start": start,
                "segment_stop": stop,
            }
            # Onsets in prepared data use segment-local coordinates, including negative
            # values when the perturbation was already active at the segment start.
            if metadata.get("perturbation_onset") is not None:
                metadata["source_perturbation_onset"] = metadata["perturbation_onset"]
                metadata["perturbation_onset"] -= start
            if isinstance(metadata.get("perturbation"), dict):
                metadata["perturbation"] = dict(metadata["perturbation"])
                if metadata["perturbation"].get("onset") is not None:
                    metadata["perturbation"]["onset"] -= start
            store.add(
                Trajectory(
                    trajectory_id=(
                        trajectory.trajectory_id
                        if len(sources) == 1
                        else f"source_{source_index}_{trajectory.trajectory_id}"
                    ),
                    true_observations=trajectory.true_observations[start : stop + 1],
                    agent_observations=trajectory.agent_observations[start : stop + 1],
                    commanded_actions=trajectory.commanded_actions[start:stop],
                    executed_actions=trajectory.executed_actions[start:stop],
                    rewards=trajectory.rewards[start:stop],
                    terminated=trajectory.terminated[start:stop],
                    truncated=trajectory.truncated[start:stop],
                    metadata=metadata,
                )
            )
            saved += 1
    if saved == 0:
        manifests[0].iloc[:0].to_parquet(store.manifest_path, index=False)
    (destination / "preparation_report.json").write_text(
        json.dumps(
            {
                **({"source_dataset": source_paths[0]} if len(sources) == 1 else {}),
                "source_datasets": source_paths,
                "episode_length": episode_length,
                "min_episode_length": min_episode_length,
                "success_threshold": success_threshold,
                "success_buffer": success_buffer,
                "episode_start": episode_start,
                "seed": seed,
                "saved": saved,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return store
