"""Reproducible artifacts: sparse embeddings, tidy units, explicit matrix axes."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    """Convert supported model, NumPy, and path values for JSON encoding.

    Args:
        value: Object the standard JSON encoder cannot handle.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    """Write indented JSON, rejecting nonfinite numbers.

    Args:
        path: Destination JSON file.
        value: Payload; supports models, dataclasses, NumPy values, and paths.
    """
    path.write_text(
        json.dumps(value, default=_json_default, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Serialize nested metadata as JSON columns, avoiding Arrow empty-struct failures.

    Args:
        path: Destination Parquet file.
        table: Rows to serialize without modifying the input table.
    """
    frame = table.copy()
    for column in frame:
        if frame[column].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            frame[column] = frame[column].map(
                lambda v: (
                    json.dumps(v, default=_json_default, sort_keys=True)
                    if isinstance(v, (dict, list, tuple))
                    else v
                )
            )
    frame.to_parquet(path, index=False)


def software_versions() -> dict[str, str]:
    """Collect Python and core dependency versions for artifact provenance."""
    result = {"python": platform.python_version()}
    for name in ("cartpole-idk", "pyidk", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def read_ids(path: Path | None, default: list[str]) -> tuple[str, ...]:
    """Read a nonempty, unique ID selection, ignoring blank lines and comments.

    Args:
        path: One-ID-per-line file, or None to use the default selection.
        default: Ordered IDs used when no file is supplied.
    """
    ids = (
        default
        if path is None
        else [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    )
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Trajectory ID selection must be nonempty and distinct")
    return tuple(ids)
