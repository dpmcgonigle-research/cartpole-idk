"""Independent fit and embed operations over selected raw trajectories."""

from __future__ import annotations

import logging
from dataclasses import replace

from cartpole_idk.artifacts import EmbeddingArtifact, FitArtifact
from cartpole_idk.idk.embedding import represented_units
from cartpole_idk.idk.fitting import fit_basis
from cartpole_idk.idk.units import UnitCollection, build_units
from cartpole_idk.model import DatasetSelection, EmbedConfig, FitConfig, WindowConfig
from cartpole_idk.storage import TrajectoryStore
from cartpole_idk.storage.embeddings import EmbeddingSet

logger = logging.getLogger(__name__)


def selected_units(selection: DatasetSelection, unit: WindowConfig) -> UnitCollection:
    """Load selected trajectories and build whole/window units with source provenance.

    Args:
        selection: Dataset path and ordered trajectory IDs to load.
        unit: Whole/window mode, transition length, and stride.
    """
    store = TrajectoryStore(selection.dataset)
    if not store.manifest_path.is_file():
        raise ValueError(f"Missing dataset manifest: {store.manifest_path}")
    trajectories = []
    for tid in selection.trajectory_ids:
        trajectory = store.get(tid)
        # Scope raw provenance to its dataset, allowing independent embedding artifacts.
        trajectory = replace(
            trajectory,
            metadata={"source_dataset": str(store.root.resolve()), **trajectory.metadata},
        )
        trajectories.append(trajectory)
    return build_units(
        trajectories, mode=unit.mode, window_length=unit.window_length, stride=unit.stride
    )


def fit_dataset(config: FitConfig) -> FitArtifact:
    """The only artifact operation that fits a scaler or IDK basis.

    Args:
        config: Training selection, unit construction, and IDK representation settings.
    """
    units = selected_units(config, config.unit)
    valid, batch = represented_units(units, config.idk)
    observation_widths = {
        getattr(u.trajectory, f"{config.idk.observation_source}_observations").shape[1]
        for u in valid.units
        if u.trajectory is not None
    }
    if len(observation_widths) != 1:
        raise ValueError("Fitting trajectories must have consistent observation widths")
    scaler, model = fit_basis(batch, config.idk)
    logger.info("Fitted %d feature rows from %d units", batch.n_samples, len(valid.units))
    return FitArtifact(
        config,
        scaler,
        model,
        valid.units,
        observation_widths.pop(),
        batch.n_samples,
        tuple(valid.skipped),
    )


def embed_dataset(fit: FitArtifact, config: EmbedConfig) -> EmbeddingArtifact:
    """Apply the saved representation, scaler and basis; never fit or resample.

    Args:
        fit: Previously saved fit artifact supplying the frozen scaler and basis.
        config: Target dataset selection, unit settings, and matching fit identity.
    """
    fit.validate()
    if config.fit_id != fit.fit_id or config.fit != fit.config:
        raise ValueError("Embedding configuration does not match the saved fit artifact")
    collection = selected_units(config, config.unit)
    for unit in collection.units:
        assert unit.trajectory is not None
        observations = (
            unit.trajectory.true_observations
            if config.fit.idk.observation_source == "true"
            else unit.trajectory.agent_observations
        )
        if observations.shape[1] != fit.observation_width:
            raise ValueError("Observation dimensionality differs from the fitted representation")
    valid, batch = represented_units(collection, fit.config.idk)
    scaled = batch.with_values(fit.scaler.transform(batch.values))
    values = fit.model.transform(scaled)
    embedded = EmbeddingSet(
        valid.units, values, fit.fit_id, fit.config.idk.t, fit.config.idk.psi, valid.skipped
    )
    logger.info("Embedded %d units with saved fit %s", len(valid.units), fit.fit_id)
    return EmbeddingArtifact(config, embedded, fit.observation_width, fit.n_fit_samples)
