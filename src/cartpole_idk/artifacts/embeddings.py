"""Sparse embeddings that can be analyzed without the original dataset or fit directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

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
from cartpole_idk.model import ArtifactMetadata, EmbedConfig
from cartpole_idk.storage.embeddings import EmbeddingSet

FILES = ("config.json", "units.parquet", "embeddings.npz")


@dataclass(slots=True)
class EmbeddingArtifact:
    """Persisted embedding population and provenance, analyzable without raw trajectories."""

    config: EmbedConfig
    embeddings: EmbeddingSet
    observation_width: int
    n_fit_samples: int
    metadata: ArtifactMetadata | None = field(default=None, init=False)

    def validate(self) -> None:
        """Check array dimensions and values against the artifact configuration."""
        embedded, idk = self.embeddings, self.config.fit.idk
        if (embedded.basis_id, embedded.t, embedded.psi) != (self.config.fit_id, idk.t, idk.psi):
            raise ValueError("Embedding dimensions/basis identity disagree with configuration")
        embedded.__post_init__()
        if not embedded.units or np.any(embedded.values.data < 0):
            raise ValueError("Embeddings require nonempty units and nonnegative mean occupancies")
        # Partition sums retain outside mass and reject count embeddings saved as means.

        columns = np.arange(idk.t * idk.psi)
        grouping = csr_matrix(
            (np.ones(len(columns)), (columns, columns // idk.psi)), shape=(len(columns), idk.t)
        )
        if np.any((embedded.values @ grouping).data > 1 + 1e-10):
            raise ValueError("Embedding partition occupancy exceeds one")

    def save(self, path: str | Path) -> None:
        """Save a versioned artifact atomically, including checksums and content identity.

        Args:
            path: New or empty destination directory.
        """
        self.validate()
        with artifact_directory(Path(path)) as root:
            (root / "config.json").write_text(
                self.config.model_dump_json(indent=2), encoding="utf-8"
            )
            save_units(
                root / "units.parquet", unit_records(self.embeddings.units, self.config.unit)
            )
            save_npz(root / "embeddings.npz", self.embeddings.values)
            files = checksums(root, FILES)
            metadata = ArtifactMetadata(
                kind="embedding",
                artifact_id=content_id(files),
                files=files,
                software=software_versions(),
                n_units=len(self.embeddings.units),
                n_features=self.embeddings.values.shape[1],
                observation_width=self.observation_width,
                n_fit_samples=self.n_fit_samples,
                parent_fit_id=self.config.fit_id,
                skipped=tuple(self.embeddings.skipped),
            )
            (root / "metadata.json").write_text(
                metadata.model_dump_json(indent=2), encoding="utf-8"
            )
        self.metadata = metadata

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingArtifact:
        """Verify and reconstruct a saved artifact without refitting or reading raw data.

        Args:
            path: Directory containing the complete artifact.
        """
        root = Path(path)
        metadata = load_metadata(root, "embedding", FILES)
        config = EmbedConfig.model_validate_json((root / "config.json").read_text(encoding="utf-8"))
        units = load_units(
            root / "units.parquet", config.unit, config.trajectory_ids, metadata.n_units
        )
        try:
            values = load_npz(root / "embeddings.npz").tocsr()
        except (TypeError, ValueError, KeyError, OSError) as exc:
            raise ValueError(f"Invalid sparse embedding payload: {exc}") from exc
        idk = config.fit.idk
        if metadata.parent_fit_id != config.fit_id or metadata.n_features != idk.t * idk.psi:
            raise ValueError("Embedding metadata is inconsistent with parent fit or dimensions")
        embedded = EmbeddingSet(
            units, values, config.fit_id, idk.t, idk.psi, list(metadata.skipped)
        )
        result = cls(config, embedded, metadata.observation_width, metadata.n_fit_samples)
        result.metadata = metadata
        result.validate()
        return result

    def compatible_with(self, other: EmbeddingArtifact) -> None:
        """Require the same fitted model and representation for cross-population analysis.

        Args:
            other: Embedding artifact to compare with this one.
        """
        self.embeddings.compatible_with(other.embeddings)
        if self.config.fit != other.config.fit or self.observation_width != other.observation_width:
            raise ValueError("Embedding artifacts have incompatible representation/fit metadata")
