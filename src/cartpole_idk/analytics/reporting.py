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
from scipy.sparse import csr_matrix, save_npz, vstack

from cartpole_idk.analytics.embeddings import AnalysisBasis, EmbeddingSet


def _json_default(value: Any) -> Any:
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
    path.write_text(
        json.dumps(value, default=_json_default, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Serialize nested metadata as JSON columns, avoiding Arrow empty-struct failures."""
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
    result = {"python": platform.python_version()}
    for name in ("cartpole-idk", "pyidk", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def write_run(
    output: Path, basis: AnalysisBasis, populations: dict[str, EmbeddingSet], config: dict[str, Any]
) -> None:
    """One deduplicated embedding file; roles and matrix axes specify row order."""
    units: list[dict[str, Any]] = []
    rows: list[csr_matrix] = []
    seen: set[str] = set()
    roles: dict[str, list[str]] = {}
    skipped: list[dict[str, Any]] = []
    for role, population in populations.items():
        roles[role] = [u.unit_id for u in population.units]
        for i, unit in enumerate(population.units):
            if unit.unit_id not in seen:
                seen.add(unit.unit_id)
                units.append({"embedding_row": len(rows), **unit.record()})
                rows.append(population.values[i])
        skipped.extend({"population": role, **record} for record in population.skipped)
    write_table(output / "units.parquet", pd.DataFrame(units))
    save_npz(output / "embeddings.npz", vstack(rows, format="csr"))
    if skipped:
        write_table(output / "skipped_units.parquet", pd.DataFrame(skipped))
    write_json(
        output / "config.json",
        {
            **config,
            "unit_roles": roles,
            "basis_id": basis.fitting.basis_id,
            "idk": asdict(basis.reference.config),
            "scaler": {
                "type": "pyidk.Standardizer",
                "mean": basis.reference.scaler.mean_,
                "scale": basis.reference.scaler.scale_,
            },
            "versions": software_versions(),
        },
    )
    fitted = basis.reference.model.point_kernel.basis_
    if fitted is not None:
        assert basis.reference.scaler.mean_ is not None
        assert basis.reference.scaler.scale_ is not None
        np.savez_compressed(
            output / "basis.npz",
            centers=fitted.centers,
            radii=fitted.radii,
            sample_indices=fitted.sample_indices,
            metric=fitted.metric,
            scaler_mean=basis.reference.scaler.mean_,
            scaler_scale=basis.reference.scaler.scale_,
        )
    write_json(
        output / "summary.json",
        {
            "unique_units": len(units),
            "skipped": len(skipped),
            "population_sizes": {k: len(v) for k, v in roles.items()},
        },
    )
