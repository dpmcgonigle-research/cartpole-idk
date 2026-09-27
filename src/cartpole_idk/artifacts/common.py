"""Versioned artifact integrity and atomic directory persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cartpole_idk.model import ArtifactMetadata, UnitRecord, WindowConfig
from cartpole_idk.storage.units import AnalysisUnit


@contextmanager
def artifact_directory(output: Path) -> Iterator[Path]:
    """Yield a staging directory and publish it atomically on successful exit.

    Args:
        output: New or empty final artifact directory.
    """
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cartpole-artifact-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        yield staging
        staging.replace(output)


def checksums(root: Path, names: Sequence[str]) -> dict[str, str]:
    """Compute SHA-256 hashes for required artifact files, rejecting missing files.

    Args:
        root: Artifact directory.
        names: Required filenames relative to root.
    """
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing required artifact file: {path}")
        with path.open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def content_id(files: dict[str, str]) -> str:
    """Derive a stable artifact identity from its file hashes and format version.

    Args:
        files: Mapping of artifact filenames to SHA-256 hashes.
    """
    return hashlib.sha256(
        json.dumps({"format_version": 1, "files": files}, sort_keys=True).encode()
    ).hexdigest()


def load_metadata(root: Path, kind: str, names: Sequence[str]) -> ArtifactMetadata:
    """Load metadata and verify artifact kind, file hashes, and content identity.

    Args:
        root: Artifact directory.
        kind: Expected artifact kind: fit or embedding.
        names: Required files whose hashes must match the metadata.
    """
    path = root / "metadata.json"
    if not path.is_file():
        raise ValueError(f"Missing required artifact file: {path}")
    metadata = ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    if metadata.kind != kind:
        raise ValueError(f"Expected {kind} artifact, got {metadata.kind}")
    actual = checksums(root, names)
    if metadata.files != actual or metadata.artifact_id != content_id(actual):
        raise ValueError("Artifact checksum/identity mismatch; files were changed or corrupted")
    return metadata


def unit_records(units: Sequence[AnalysisUnit], config: WindowConfig) -> list[UnitRecord]:
    """Convert ordered units to validated records for persistence.

    Args:
        units: Units in embedding-row order.
        config: Unit construction settings supplying the whole/window mode.
    """
    return [
        UnitRecord(
            unit_id=u.unit_id,
            trajectory_id=u.trajectory_id,
            start_step=u.start_step,
            end_step=u.end_step,
            mode=config.mode,
            metadata=u.metadata,
        )
        for u in units
    ]


def save_units(path: Path, records: list[UnitRecord]) -> None:
    # Exact nested metadata round-trips without Arrow struct inference or NaN coercion.
    """Write ordered unit records to Parquet, preserving nested metadata as JSON.

    Args:
        path: Destination Parquet file.
        records: Unit provenance records in embedding-row order.
    """
    rows = [
        {
            "embedding_row": i,
            "unit_id": r.unit_id,
            "trajectory_id": r.trajectory_id,
            "mode": r.mode,
            "start_step": r.start_step,
            "end_step": r.end_step,
            "raw_length": r.end_step - r.start_step,
            "metadata_json": json.dumps(r.metadata, sort_keys=True, allow_nan=False),
        }
        for i, r in enumerate(records)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_units(
    path: Path, config: WindowConfig, selected_ids: tuple[str, ...], n_units: int
) -> list[AnalysisUnit]:
    """Validate saved unit provenance and reconstruct units without raw arrays.

    Args:
        path: Saved unit-manifest Parquet file.
        config: Expected unit mode, transition length, and stride.
        selected_ids: Trajectory IDs permitted by the artifact configuration.
        n_units: Expected number of rows.
    """
    frame = pd.read_parquet(path)
    required = {
        "embedding_row",
        "unit_id",
        "trajectory_id",
        "start_step",
        "end_step",
        "mode",
        "raw_length",
        "metadata_json",
    }
    if not required <= set(frame) or len(frame) != n_units:
        raise ValueError("Unit manifest has missing columns or an inconsistent row count")
    if not np.array_equal(frame.embedding_row.to_numpy(), np.arange(n_units)):
        raise ValueError("Unit manifest row ordering does not match embeddings")
    units = []
    for row in frame.to_dict("records"):
        record = UnitRecord.model_validate(
            {
                key: row[key]
                for key in ("unit_id", "trajectory_id", "start_step", "end_step", "mode")
            }
            | {"metadata": json.loads(row["metadata_json"])}
        )
        length = record.end_step - record.start_step
        if (
            record.mode != config.mode
            or record.trajectory_id not in selected_ids
            or length != row["raw_length"]
        ):
            raise ValueError("Unit provenance does not match artifact configuration")
        if config.mode == "window" and (
            length != config.window_length or record.start_step % config.stride
        ):
            raise ValueError("Window interval does not match embedding configuration")
        if config.mode == "whole" and record.start_step != 0:
            raise ValueError("Whole-trajectory unit must start at zero")
        stored_length = record.metadata.get("stored_length", record.end_step)
        if not isinstance(stored_length, int) or record.end_step > stored_length:
            raise ValueError("Unit interval exceeds its recorded trajectory length")
        units.append(
            AnalysisUnit(
                record.unit_id,
                record.trajectory_id,
                record.start_step,
                record.end_step,
                dict(record.metadata),
            )
        )
    if len({u.unit_id for u in units}) != len(units):
        raise ValueError("Unit IDs must be unique")
    return units
