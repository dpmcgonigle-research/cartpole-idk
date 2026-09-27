"""Explicit NumPy persistence and reconstruction of a fitted pyidk model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyidk import IsolationDistributionalKernel, IsolationKernel, Standardizer
from pyidk.partition import IsolationBasis

from cartpole_idk.artifacts.common import (
    artifact_directory,
    checksums,
    content_id,
    load_metadata,
    load_units,
    save_units,
    unit_records,
)
from cartpole_idk.artifacts.io import software_versions
from cartpole_idk.model import ArtifactMetadata, FitConfig
from cartpole_idk.storage.units import AnalysisUnit

FILES = ("config.json", "scaler.npz", "basis.npz", "fit_manifest.parquet")


@dataclass(slots=True)
class FitArtifact:
    """Reusable fitted scaler and isolation basis with configuration and training provenance."""

    config: FitConfig
    scaler: Standardizer
    model: IsolationDistributionalKernel
    units: list[AnalysisUnit]
    observation_width: int
    n_fit_samples: int
    skipped: tuple[dict[str, object], ...] = ()
    metadata: ArtifactMetadata | None = field(default=None, init=False)

    @property
    def fit_id(self) -> str:
        """Content identity assigned by saving or loading this artifact."""
        if self.metadata is None:
            raise ValueError("Save the fit artifact before creating persistent embeddings")
        return self.metadata.artifact_id

    def validate(self) -> None:
        """Check array dimensions and values against the artifact configuration."""
        basis = self.model.point_kernel.basis_
        mean, scale = self.scaler.mean_, self.scaler.scale_
        if basis is None or mean is None or scale is None:
            raise ValueError("Fit artifact requires a fitted scaler and basis")
        idk = self.config.idk
        width = idk.feature_width(self.observation_width)
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("Scaler dimensionality is incompatible with representation")
        if basis.centers.shape != (idk.t, idk.psi, width):
            raise ValueError("Basis dimensionality is incompatible with scaler/IDK configuration")
        if basis.radii.shape != (idk.t, idk.psi) or basis.sample_indices.shape != (idk.t, idk.psi):
            raise ValueError("Invalid basis radii or sample-index dimensions")
        if not all(np.isfinite(x).all() for x in (mean, scale, basis.centers, basis.radii)):
            raise ValueError("Scaler and basis arrays must be finite")
        if np.any(scale <= 0) or np.any(basis.radii < 0):
            raise ValueError("Scaler scales must be positive and radii nonnegative")
        indices = basis.sample_indices
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= self.n_fit_samples)
            or self.n_fit_samples < idk.psi
        ):
            raise ValueError("Basis sample indices are inconsistent with the fitting population")
        if basis.metric != "euclidean":
            raise ValueError("Unsupported basis metric; expected euclidean")
        if (
            self.model.point_kernel.n_partitions,
            self.model.point_kernel.samples_per_partition,
        ) != (idk.t, idk.psi):
            raise ValueError("Model parameters disagree with saved IDK configuration")
        if not self.units:
            raise ValueError("Fit artifact must contain fitting provenance")

    def save(self, path: str | Path) -> None:
        """Save a versioned artifact atomically, including checksums and content identity.

        Args:
            path: New or empty destination directory.
        """
        self.validate()
        basis = self.model.point_kernel.basis_
        assert (
            basis is not None and self.scaler.mean_ is not None and self.scaler.scale_ is not None
        )
        with artifact_directory(Path(path)) as root:
            (root / "config.json").write_text(
                self.config.model_dump_json(indent=2), encoding="utf-8"
            )
            np.savez_compressed(
                root / "scaler.npz", mean=self.scaler.mean_, scale=self.scaler.scale_
            )
            np.savez_compressed(
                root / "basis.npz",
                centers=basis.centers,
                radii=basis.radii,
                sample_indices=basis.sample_indices,
                metric=np.asarray(basis.metric),
            )
            save_units(root / "fit_manifest.parquet", unit_records(self.units, self.config.unit))
            files = checksums(root, FILES)
            metadata = ArtifactMetadata(
                kind="fit",
                artifact_id=content_id(files),
                files=files,
                software=software_versions(),
                n_units=len(self.units),
                observation_width=self.observation_width,
                n_features=len(self.scaler.mean_),
                n_fit_samples=self.n_fit_samples,
                skipped=self.skipped,
            )
            (root / "metadata.json").write_text(
                metadata.model_dump_json(indent=2), encoding="utf-8"
            )
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path) -> FitArtifact:
        """Verify and reconstruct a saved artifact without refitting or reading raw data.

        Args:
            path: Directory containing the complete artifact.
        """
        root = Path(path)
        metadata = load_metadata(root, "fit", FILES)
        config = FitConfig.model_validate_json((root / "config.json").read_text(encoding="utf-8"))
        try:
            with np.load(root / "scaler.npz", allow_pickle=False) as data:
                scaler = Standardizer(mean_=data["mean"], scale_=data["scale"])
            with np.load(root / "basis.npz", allow_pickle=False) as data:
                basis = IsolationBasis(
                    centers=data["centers"],
                    radii=data["radii"],
                    sample_indices=data["sample_indices"],
                    metric=str(data["metric"].item()),
                )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid fit array payload: {exc}") from exc
        point_kernel = IsolationKernel(
            n_partitions=config.idk.t,
            samples_per_partition=config.idk.psi,
            random_state=config.idk.random_state,
        )
        point_kernel.basis_ = basis
        units = load_units(
            root / "fit_manifest.parquet", config.unit, config.trajectory_ids, metadata.n_units
        )
        result = cls(
            config,
            scaler,
            IsolationDistributionalKernel(point_kernel),
            units,
            metadata.observation_width,
            metadata.n_fit_samples,
            metadata.skipped,
        )
        result.metadata = metadata
        result.validate()
        if metadata.n_features != basis.n_features_in or metadata.parent_fit_id is not None:
            raise ValueError("Fit metadata is inconsistent with the saved basis")
        return result
