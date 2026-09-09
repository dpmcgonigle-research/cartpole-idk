from __future__ import annotations

from dataclasses import dataclass

from pyidk import IsolationDistributionalKernel, IsolationKernel, Standardizer

from cartpole_idk.storage import TrajectoryStore

from .config import IDKExperimentConfig
from .features import build_sequence_batch


@dataclass(slots=True)
class IDKReference:
    config: IDKExperimentConfig
    scaler: Standardizer
    model: IsolationDistributionalKernel
    reference_embeddings: object
    trajectory_ids: list[str]


def fit_reference(
    store: TrajectoryStore, trajectory_ids: list[str], config: IDKExperimentConfig
) -> IDKReference:
    trajectories = [store.get(tid) for tid in trajectory_ids]
    batch = build_sequence_batch(trajectories, config)
    scaler = Standardizer().fit(batch.values)
    scaled = batch.with_values(scaler.transform(batch.values))
    point_kernel = IsolationKernel(
        n_partitions=config.t,
        samples_per_partition=config.psi,
        random_state=config.random_state,
    )
    model = IsolationDistributionalKernel(point_kernel)
    embeddings = model.fit_transform(scaled)
    return IDKReference(config, scaler, model, embeddings, list(trajectory_ids))
