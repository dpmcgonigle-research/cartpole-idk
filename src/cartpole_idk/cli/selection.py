"""Load trajectory ID selections for dataset commands."""

from pathlib import Path

from cartpole_idk.artifacts.io import read_ids
from cartpole_idk.storage import TrajectoryStore


def selection(dataset: Path, ids_file: Path | None) -> tuple[str, ...]:
    """Select IDs from a file or use the dataset's full manifest order.

    Args:
        dataset: Trajectory dataset directory.
        ids_file: Optional one-ID-per-line selection file.
    """
    store = TrajectoryStore(dataset)
    if not store.manifest_path.is_file():
        raise ValueError(f"Missing dataset manifest: {store.manifest_path}")
    return read_ids(ids_file, store.manifest().trajectory_id.tolist())
