"""Analytics over saved embeddings only: no raw data, fitting or transformation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cartpole_idk.analytics.clustering import cluster_units
from cartpole_idk.analytics.evaluation import cluster_summary, evaluate_clusters
from cartpole_idk.analytics.metrics import METRICS, pairwise
from cartpole_idk.analytics.neighbors import reference_likeness, top_k_neighbors
from cartpole_idk.analytics.population import population_mmd
from cartpole_idk.analytics.reporting import software_versions, write_json, write_table
from cartpole_idk.artifacts import EmbeddingArtifact
from cartpole_idk.artifacts.common import artifact_directory
from cartpole_idk.model import AnalysisConfig, AnalysisInput, AnalysisMetadata
from cartpole_idk.storage.embeddings import EmbeddingSet

logger = logging.getLogger(__name__)


def execute_analysis(config: AnalysisConfig) -> None:
    """Load compatible artifacts and perform exactly one typed analysis operation."""
    cache: dict[Path, EmbeddingArtifact] = {}
    inputs: dict[str, AnalysisInput] = {}
    populations: dict[str, EmbeddingSet] = {}

    def load(role: str, path: Path) -> EmbeddingSet:
        path = path.resolve()
        if path not in cache:
            cache[path] = EmbeddingArtifact.load(path)
        artifact = cache[path]
        if cache:
            next(iter(cache.values())).compatible_with(artifact)
        assert artifact.metadata is not None
        inputs[role] = AnalysisInput(
            path=path, artifact_id=artifact.metadata.artifact_id, fit_id=artifact.config.fit_id
        )
        populations[role] = artifact.embeddings
        return artifact.embeddings

    analysis = load("analysis", config.embeddings)
    if config.command == "rolling":
        if cache[config.embeddings.resolve()].config.unit.mode != "window":
            raise ValueError("Rolling analysis requires an existing window embedding artifact")
        indices = [i for i, u in enumerate(analysis.units) if u.trajectory_id == config.query_id]
        if not indices:
            raise ValueError(f"No embedded windows for query trajectory {config.query_id}")
        analysis = analysis.subset(indices)
        populations["analysis"] = analysis
    reference = None
    nominal = failure = group_b = None
    if config.nominal_reference is not None and config.failure_reference is not None:
        nominal = load("nominal", config.nominal_reference)
        failure = load("failure", config.failure_reference)
    elif config.command in {"neighbors", "rolling"} or config.reference is not None:
        reference = load("reference", config.reference or config.embeddings)
    if config.group_b is not None:
        group_b = load("group_b", config.group_b)
    metric_parameters = {"epsilon": config.metric.epsilon} if config.metric.metric == "kl" else {}
    with artifact_directory(config.output) as output:
        if config.command == "pairwise":
            matrix = pairwise(
                analysis,
                reference,
                metric=config.metric.metric,
                max_pairs=config.max_pairs,
                **metric_parameters,
            )
            np.save(output / "pairwise.npy", matrix)
            write_json(
                output / "pairwise_axes.json",
                {
                    "rows": [u.unit_id for u in analysis.units],
                    "columns": [u.unit_id for u in (reference or analysis).units],
                },
            )
        elif config.command in {"neighbors", "rolling"}:
            if nominal is not None and failure is not None:
                scores, results = reference_likeness(
                    analysis,
                    nominal,
                    failure,
                    metric=config.metric.metric,
                    max_pairs=config.max_pairs,
                    k=config.neighbors.k,
                    exclude_same_unit=not config.neighbors.include_self,
                    exclude_same_trajectory=config.neighbors.exclude_same_trajectory,
                    max_overlap=config.neighbors.max_overlap,
                    metric_parameters=metric_parameters,
                )
                for name, result in results.items():
                    write_table(output / f"{name}_neighbors.parquet", result.neighbors)
            else:
                assert reference is not None
                result = top_k_neighbors(
                    analysis,
                    reference,
                    metric=config.metric.metric,
                    max_pairs=config.max_pairs,
                    k=config.neighbors.k,
                    exclude_same_unit=not config.neighbors.include_self,
                    exclude_same_trajectory=config.neighbors.exclude_same_trajectory,
                    max_overlap=config.neighbors.max_overlap,
                    metric_parameters=metric_parameters,
                )
                scores = result.scores
                write_table(output / "neighbors.parquet", result.neighbors)
            write_table(
                output / ("rolling.parquet" if config.command == "rolling" else "scores.parquet"),
                scores,
            )
        elif config.command == "cluster":
            result_cluster = cluster_units(analysis, config.cluster, max_pairs=config.max_pairs)
            write_table(output / "cluster_assignments.parquet", result_cluster.assignments)
            summary, composition = cluster_summary(
                result_cluster.assignments, categorical=config.cluster.categorical
            )
            write_table(output / "cluster_summary.parquet", summary)
            write_table(output / "cluster_composition.parquet", composition)
            write_json(
                output / "cluster_evaluation.json",
                evaluate_clusters(
                    result_cluster.assignments,
                    result_cluster.distances,
                    categorical=config.cluster.categorical,
                ),
            )
            if result_cluster.coordinates is not None:
                np.save(output / "coordinates.npy", result_cluster.coordinates)
            if result_cluster.probabilities is not None:
                np.save(output / "cluster_probabilities.npy", result_cluster.probabilities)
            if result_cluster.components:
                np.savez_compressed(output / "components.npz", **result_cluster.components)
        elif config.command == "population":
            assert group_b is not None
            mmd = population_mmd(
                analysis,
                group_b,
                bandwidth=config.population.bandwidth,
                estimator=config.population.estimator,
                n_permutations=config.population.permutations,
                random_state=config.population.random_state,
                max_pairs=config.max_pairs,
            )
            write_json(output / "population.json", mmd)
        if config.command in {"pairwise", "neighbors", "rolling"}:
            spec = METRICS[config.metric.metric]
            write_json(
                output / "metric.json",
                {
                    **config.metric.model_dump(mode="json"),
                    "direction": spec.direction,
                    "description": spec.description,
                },
            )
        for role, population in populations.items():
            filename = "units.parquet" if role == "analysis" else f"{role}_units.parquet"
            write_table(output / filename, pd.DataFrame([u.record() for u in population.units]))
        (output / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
        metadata = AnalysisMetadata(
            inputs=inputs,
            software=software_versions(),
            unit_roles={
                role: tuple(u.unit_id for u in value.units) for role, value in populations.items()
            },
        )
        (output / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Analysis complete: %s", config.output)
