from __future__ import annotations

from dataclasses import dataclass

from pyidk import IsolationDistributionalKernel, IsolationKernel, SequenceBatch, Standardizer
from scipy.sparse import csr_matrix

from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.idk.features import build_sequence_batch
from cartpole_idk.storage import TrajectoryStore


@dataclass(slots=True)
class IDKReference:
    config: IDKExperimentConfig
    scaler: Standardizer
    model: IsolationDistributionalKernel
    reference_embeddings: csr_matrix
    trajectory_ids: list[str]


def fit_reference(
    store: TrajectoryStore, trajectory_ids: list[str], config: IDKExperimentConfig
) -> IDKReference:
    trajectories = [store.get(tid) for tid in trajectory_ids]
    batch = build_sequence_batch(trajectories, config)
    return fit_sequence_batch(batch, trajectory_ids, config)


def fit_sequence_batch(
    batch: SequenceBatch, sequence_ids: list[str], config: IDKExperimentConfig
) -> IDKReference:
    """Fit scaler and pyidk basis exclusively on the supplied fitting sequences."""
    if len(sequence_ids) != batch.n_sequences or (batch.lengths == 0).any():
        raise ValueError("Provide one ID per nonempty fitting sequence")
    scaler, model = fit_basis(batch, config)
    scaled = batch.with_values(scaler.transform(batch.values))
    embeddings = model.transform(scaled)
    return IDKReference(config, scaler, model, embeddings, list(sequence_ids))


def fit_basis(
    batch: SequenceBatch, config: IDKExperimentConfig
) -> tuple[Standardizer, IsolationDistributionalKernel]:
    """Fit the reusable scaler and basis without generating distribution embeddings."""
    scaler = Standardizer().fit(batch.values)
    scaled = batch.with_values(scaler.transform(batch.values))
    point_kernel = IsolationKernel(
        n_partitions=config.t, samples_per_partition=config.psi, random_state=config.random_state
    )
    model = IsolationDistributionalKernel(point_kernel).fit(scaled)
    return scaler, model
