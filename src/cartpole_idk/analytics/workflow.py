"""Reusable run orchestration, independent of the CLI parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.sparse import vstack

from cartpole_idk.analytics.clustering import cluster_units
from cartpole_idk.analytics.embeddings import (
    AnalysisBasis,
    EmbeddingSet,
    fit_embeddings,
    represented_units,
)
from cartpole_idk.analytics.evaluation import cluster_summary, evaluate_clusters
from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, METRICS, pairwise
from cartpole_idk.analytics.neighbors import reference_likeness, top_k_neighbors
from cartpole_idk.analytics.population import population_mmd
from cartpole_idk.analytics.reporting import write_json, write_run, write_table
from cartpole_idk.analytics.units import UnitCollection, build_units
from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.storage import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    dataset: Path
    output: Path
    mode: Literal["whole", "window"] = "whole"
    window_length: int = 25
    stride: int = 1
    idk: IDKExperimentConfig = field(default_factory=IDKExperimentConfig)
    max_pairs: int = DEFAULT_MAX_PAIRS


def read_ids(path: Path | None, default: list[str]) -> list[str]:
    """One ID per nonblank, noncomment line; reject duplicate/empty explicit selections."""
    if path is None:
        return list(default)
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file must contain nonempty, distinct IDs: {path}")
    return ids


class AnalysisRun:
    """One fitted basis and a cache of transformed units shared by run populations."""

    def __init__(self, config: AnalysisConfig, fit_ids: list[str]):
        self.config = config
        self.store = TrajectoryStore(config.dataset)
        self.selections: dict[str, list[str]] = {"fit": fit_ids}
        self.basis: AnalysisBasis = fit_embeddings(self._units(fit_ids), config.idk)
        self.populations = {"fit": self.basis.fitting}
        self.cache = {
            u.unit_id: (u, self.basis.fitting.values[i])
            for i, u in enumerate(self.basis.fitting.units)
        }
        logger.info("Fitted scaler and IDK basis on %d units", len(self.basis.fitting.units))

    def _units(self, ids: list[str]) -> UnitCollection:
        return build_units(
            [self.store.get(tid) for tid in ids],
            mode=self.config.mode,
            window_length=self.config.window_length,
            stride=self.config.stride,
        )

    def embed(self, role: str, ids: list[str]) -> EmbeddingSet:
        self.selections[role] = ids
        collection, _ = represented_units(self._units(ids), self.config.idk)
        unseen = [u for u in collection.units if u.unit_id not in self.cache]
        skipped = list(collection.skipped)
        if unseen:
            embedded = self.basis.transform(UnitCollection(unseen))
            skipped.extend(embedded.skipped)
            self.cache.update(
                (u.unit_id, (u, embedded.values[i])) for i, u in enumerate(embedded.units)
            )
        units = [u for u in collection.units if u.unit_id in self.cache]
        if not units:
            raise ValueError(f"No valid units for {role}")
        values = vstack([self.cache[u.unit_id][1] for u in units], format="csr")
        result = EmbeddingSet(
            units,
            values,
            self.basis.fitting.basis_id,
            self.config.idk.t,
            self.config.idk.psi,
            skipped,
        )
        self.populations[role] = result
        return result

    def save(self, options: dict[str, Any]) -> None:
        write_run(
            self.config.output,
            self.basis,
            self.populations,
            {"analysis": self.config, "trajectory_roles": self.selections, **options},
        )


def execute_analysis(
    command: str,
    config: AnalysisConfig,
    *,
    selections: dict[str, Path | None],
    options: dict[str, Any],
) -> None:
    """Dispatch one analysis task; validate explicit role selection before writing."""
    source = TrajectoryStore(config.dataset)
    if not source.manifest_path.exists():
        raise ValueError(f"No dataset manifest at {source.manifest_path}")
    all_ids = source.manifest().trajectory_id.tolist()
    fit_ids = read_ids(selections.get("fit"), all_ids)
    analysis_ids = read_ids(selections.get("analysis"), all_ids)
    reference_ids = read_ids(selections.get("reference"), fit_ids)
    resolved = {name: read_ids(path, []) for name, path in selections.items() if path is not None}
    if command == "rolling":
        if not options.get("query_id"):
            raise ValueError("rolling requires --query-id")
        analysis_ids = [options["query_id"]]
    if command == "population" and not all(name in resolved for name in ("group_a", "group_b")):
        raise ValueError("population requires --group-a-ids-file and --group-b-ids-file")
    if ("nominal" in resolved) != ("failure" in resolved):
        raise ValueError("Provide both nominal and failure reference ID files")
    known = set(all_ids)
    for ids in [fit_ids, analysis_ids, reference_ids, *resolved.values()]:
        if not set(ids) <= known:
            raise ValueError(f"Unknown trajectory IDs: {sorted(set(ids) - known)}")
    output = config.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    run = AnalysisRun(config, fit_ids)
    output.mkdir(parents=True, exist_ok=True)
    metric = options.get("metric", "idk")
    metric_parameters = {"epsilon": options["epsilon"]} if metric == "kl" else {}
    common = {"metric": metric, "max_pairs": config.max_pairs}
    if command != "population":
        analysis = run.embed("analysis", analysis_ids)
    if command == "pairwise":
        matrix = pairwise(analysis, **common, **metric_parameters)
        np.save(output / "pairwise.npy", matrix)
        ids = [u.unit_id for u in analysis.units]
        write_json(output / "pairwise_axes.json", {"rows": ids, "columns": ids})
    elif command in {"neighbors", "rolling"}:
        neighbor_options = {
            **common,
            "k": options["k"],
            "exclude_same_unit": not options.get("include_self", False),
            "exclude_same_trajectory": options.get("exclude_same_trajectory", False),
            "max_overlap": options.get("max_overlap"),
            "metric_parameters": metric_parameters,
        }
        if "nominal" in resolved:
            nominal, failure = (
                run.embed("nominal", resolved["nominal"]),
                run.embed("failure", resolved["failure"]),
            )
            scores, results = reference_likeness(analysis, nominal, failure, **neighbor_options)
            for name, result in results.items():
                write_table(output / f"{name}_neighbors.parquet", result.neighbors)
        else:
            reference = run.embed("reference", reference_ids)
            result = top_k_neighbors(analysis, reference, **neighbor_options)
            scores = result.scores
            write_table(output / "neighbors.parquet", result.neighbors)
        write_table(
            output / ("rolling.parquet" if command == "rolling" else "scores.parquet"), scores
        )
    elif command == "cluster":
        cluster_config = options["cluster_config"]
        result_cluster = cluster_units(analysis, cluster_config, max_pairs=config.max_pairs)
        write_table(output / "cluster_assignments.parquet", result_cluster.assignments)
        categories = tuple(options.get("categorical", ["perturbation_type", "checkpoint_id"]))
        summary, composition = cluster_summary(result_cluster.assignments, categorical=categories)
        write_table(output / "cluster_summary.parquet", summary)
        write_table(output / "cluster_composition.parquet", composition)
        write_json(
            output / "cluster_evaluation.json",
            evaluate_clusters(
                result_cluster.assignments, result_cluster.distances, categorical=categories
            ),
        )
        if result_cluster.coordinates is not None:
            np.save(output / "coordinates.npy", result_cluster.coordinates)
        if result_cluster.probabilities is not None:
            np.save(output / "cluster_probabilities.npy", result_cluster.probabilities)
        if result_cluster.components:
            np.savez_compressed(output / "components.npz", **result_cluster.components)
    elif command == "population":
        a, b = run.embed("group_a", resolved["group_a"]), run.embed("group_b", resolved["group_b"])
        mmd = population_mmd(
            a,
            b,
            bandwidth=options.get("bandwidth"),
            estimator=options.get("estimator", "biased"),
            n_permutations=options.get("permutations", 0),
            random_state=options["random_state"],
            max_pairs=config.max_pairs,
        )
        write_json(output / "population.json", mmd)
    else:
        raise ValueError(f"Unknown analysis command: {command}")
    if command in {"pairwise", "neighbors", "rolling"}:
        write_json(
            output / "metric.json",
            {
                "name": metric,
                "direction": METRICS[metric].direction,
                "description": METRICS[metric].description,
                "parameters": metric_parameters,
            },
        )
    run.save({"command": command, "options": options, "selection_files": selections})
    logger.info("Analysis complete: %s", output)
