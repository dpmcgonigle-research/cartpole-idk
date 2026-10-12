"""Moving-window analytics on reusable pyidk point features, without plotting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyidk import PointFeatureCache, SequenceBatch, reduce_references

from cartpole_idk.analytics.metrics import PYIDK_METRICS
from cartpole_idk.analytics.neighbors import eligible_references
from cartpole_idk.artifacts import FitArtifact
from cartpole_idk.artifacts.common import artifact_directory
from cartpole_idk.artifacts.io import software_versions, write_json, write_table
from cartpole_idk.idk.features import build_sequence_batch
from cartpole_idk.idk.pipeline import selected_units
from cartpole_idk.model import DatasetSelection, WindowConfig, WindowScoringConfig
from cartpole_idk.storage.units import AnalysisUnit

SCORE_COLUMNS = [
    "analysis_id",
    "fit_id",
    "window_length",
    "unit_id",
    "trajectory_id",
    "source_dataset",
    "source_trajectory_id",
    "start_step",
    "end_step",
    "source_start_step",
    "source_end_step",
    "row_count",
    "metric",
    "reference_group",
    "aggregation",
    "score",
    "eligible_reference_count",
    "selected_count",
]
REFERENCE_COLUMNS = [
    "analysis_id",
    "fit_id",
    "window_length",
    "reference_group",
    "unit_id",
    "trajectory_id",
    "source_dataset",
    "source_trajectory_id",
    "start_step",
    "end_step",
    "source_start_step",
    "source_end_step",
    "row_count",
]


def summarize_window_scores(
    scores: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = ("source_dataset",),
    endpoint: str = "end_step",
) -> pd.DataFrame:
    """Summarize query scores at each endpoint; IQR is across stored query trajectories.

    Reference windows have already been reduced to one score per query window.
    Missing scores contribute to n_windows but not n_scored or the quantiles.
    Each stored trajectory contributes at most one window at a given local endpoint.
    Use source_end_step for original coordinates of prepared segments. These
    descriptive bands are not confidence intervals; adjacent windows overlap.
    """
    keys = list(
        dict.fromkeys(
            [
                "analysis_id",
                "fit_id",
                "window_length",
                "metric",
                "reference_group",
                "aggregation",
                *group_by,
                endpoint,
            ]
        )
    )
    if any(key not in scores for key in keys):
        raise ValueError("Summary grouping and endpoint columns must exist in scores")
    if scores.empty:
        return pd.DataFrame(
            columns=[*keys, "n_windows", "n_scored", "mean", "median", "q25", "q75"]
        )
    return (
        scores.groupby(keys, dropna=False, sort=False, observed=True)["score"]
        .agg(
            n_windows="size",
            n_scored="count",
            mean="mean",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


@dataclass(slots=True)
class WindowAnalyticsResult:
    """Tidy scores, descriptive endpoint summaries, and reproducible reference selections."""

    scores: pd.DataFrame
    summary: pd.DataFrame
    reference_windows: pd.DataFrame
    skipped: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output: str | Path) -> None:
        """Write tables and full settings atomically to a new or empty directory."""
        with artifact_directory(Path(output)) as destination:
            write_table(destination / "scores.parquet", self.scores)
            write_table(destination / "summary.parquet", self.summary)
            write_table(destination / "reference_windows.parquet", self.reference_windows)
            write_table(destination / "skipped.parquet", self.skipped)
            write_json(destination / "metadata.json", self.metadata)


@dataclass(slots=True)
class _Population:
    units: list[AnalysisUnit]
    cache: PointFeatureCache
    reward_sums: list[np.ndarray]
    skipped: list[dict[str, Any]]

    def unit(self, interval: np.ndarray | tuple[int, int, int], length: int) -> AnalysisUnit:
        sequence, start, stop = map(int, interval)
        whole = self.units[sequence]
        end = start + length
        offset = whole.metadata["source_start_step"]
        onset = whole.metadata.get("perturbation_onset")
        terminal = whole.metadata["terminated"] or whole.metadata["truncated"]
        # Keep the same ID convention as persisted window embeddings.
        prefix = whole.unit_id.rsplit(":", 2)[0]
        return AnalysisUnit(
            f"{prefix}:{start}:{end}",
            whole.trajectory_id,
            start,
            end,
            {
                **whole.metadata,
                "mode": "window",
                "source_trajectory_id": whole.source_key[1],
                "source_start_step": offset + start,
                "source_end_step": offset + end,
                "segment_return": float(
                    self.reward_sums[sequence][end] - self.reward_sums[sequence][start]
                ),
                "time_until_termination": whole.raw_length - end if terminal else None,
                "time_until_failure": (
                    whole.raw_length - end if whole.metadata["terminated"] else None
                ),
                "time_since_perturbation": end - onset if onset is not None else None,
                "row_count": stop - start,
            },
        )


class WindowAnalytics:
    """Transform each selected population once using one saved scaler and basis.

    Reuse this session's score method for window-size/stride/metric sweeps.
    Create a new session when changing the fit, scaler, representation, or data.
    Query windows stream in batches; the selected reference bank stays sparse
    in memory. No query-reference timestep alignment or temporal matching occurs.
    Treat the fit and session caches as immutable while using the session.
    """

    def __init__(
        self,
        fit: FitArtifact,
        query: DatasetSelection,
        references: Mapping[str, DatasetSelection],
    ) -> None:
        fit.validate()
        self._fit_id = fit.fit_id
        if any(not name or name == "__basis__" for name in references):
            raise ValueError("Reference group names must be nonempty; __basis__ is reserved")
        self.fit = fit
        self.query_selection = query.model_copy(update={"dataset": query.dataset.resolve()})
        self.reference_selections = {
            name: selection.model_copy(update={"dataset": selection.dataset.resolve()})
            for name, selection in references.items()
        }
        self._fit_config = fit.config.model_dump_json()
        # Also detect mutation of preprocessing, which pyidk's basis binding cannot detect.
        self._scaler_state = self._scaler_signature()
        populations: dict[tuple[Path, tuple[str, ...]], _Population] = {}

        def prepare(selection: DatasetSelection) -> _Population:
            key = selection.dataset.resolve(), selection.trajectory_ids
            if key not in populations:
                collection = selected_units(selection, WindowConfig())
                for unit in collection.units:
                    assert unit.trajectory is not None
                    observations = getattr(
                        unit.trajectory, f"{fit.config.idk.observation_source}_observations"
                    )
                    if observations.shape[1] != fit.observation_width:
                        raise ValueError("Observation dimensionality differs from the saved fit")
                trajectories = [u.trajectory for u in collection.units if u.trajectory is not None]
                if trajectories:
                    batch = build_sequence_batch(trajectories, fit.config.idk)
                else:
                    # pyidk requires at least one sequence; an empty sequence yields no windows.
                    width = fit.config.idk.feature_width(fit.observation_width)
                    batch = SequenceBatch.from_sequences([np.empty((0, width))])
                skipped = list(collection.skipped)
                skipped.extend(
                    {
                        "trajectory_id": unit.trajectory_id,
                        "source_dataset": unit.source_key[0],
                        "reason": "empty_representation",
                    }
                    for unit, size in zip(collection.units, batch.lengths, strict=False)
                    if size == 0
                )
                scaled = (
                    batch.with_values(fit.scaler.transform(batch.values))
                    if batch.n_samples
                    else batch
                )
                cache = PointFeatureCache.from_sequences(fit.model.point_kernel, scaled)
                reward_sums = [
                    np.r_[0.0, np.cumsum(unit.trajectory.rewards, dtype=float)]
                    for unit in collection.units
                    if unit.trajectory is not None
                ]
                populations[key] = _Population(collection.units, cache, reward_sums, skipped)
            return populations[key]

        self.query = prepare(query)
        self.references = {name: prepare(selection) for name, selection in references.items()}

    def _scaler_signature(self) -> str:
        # Standardizer's public fitted parameters define the transform.
        digest = hashlib.sha256()
        for array in (self.fit.scaler.mean_, self.fit.scaler.scale_):
            values = np.asarray(array)
            digest.update(str((values.shape, values.dtype)).encode())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _row_length(self, transitions: int) -> int:
        representation = self.fit.config.idk.representation
        if representation == "state":
            return transitions + 1
        if representation == "window":
            length = transitions + 2 - self.fit.config.idk.window_length
            if length < 1:
                raise ValueError("Analysis window is too short for the temporal representation")
            return length
        return transitions

    def score(self, config: WindowScoringConfig | None = None) -> WindowAnalyticsResult:
        """Score complete current windows against references sampled anywhere in each sequence.

        mean equally weights eligible reference windows. top_k averages the best
        k eligible windows over the entire bank, with deterministic tie order.
        The optional per-trajectory cap samples without replacement from the
        reference-stride grid, using an explicit seed; shorter trajectories may
        contribute fewer windows. Same-source exclusions precede reduction.
        Outside mass is the query occupancy missing from the basis, averaged
        over partitions; it has no reference population or neighbor reduction.
        """
        config = config or WindowScoringConfig()
        if (
            self.fit.config.model_dump_json() != self._fit_config
            or self._scaler_signature() != self._scaler_state
            or self.fit.fit_id != self._fit_id
        ):
            raise ValueError("Fit configuration or scaler changed; create a new analytics session")
        self.query.cache.assert_compatible(self.fit.model.point_kernel)
        for population in self.references.values():
            self.query.cache.assert_compatible(population.cache)
        if any(metric != "outside-mass" for metric in config.metrics) and not self.references:
            raise ValueError("Pairwise window metrics require at least one reference population")
        row_length = self._row_length(config.window_length)
        metadata = {
            "fit_id": self.fit.fit_id,
            "fit": self.fit.config.model_dump(mode="json"),
            "query": self.query_selection.model_dump(mode="json"),
            "references": {
                name: selection.model_dump(mode="json")
                for name, selection in self.reference_selections.items()
            },
            "config": config.model_dump(mode="json"),
        }
        analysis_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        common = {
            "analysis_id": analysis_id,
            "fit_id": self.fit.fit_id,
            "window_length": config.window_length,
        }
        banks = {}
        reference_rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for role, population in [("query", self.query), *self.references.items()]:
            skipped.extend({"population": role, **row} for row in population.skipped)
            skipped.extend(
                {
                    "population": role,
                    "trajectory_id": unit.trajectory_id,
                    "source_dataset": unit.source_key[0],
                    "reason": "shorter_than_window",
                }
                for unit in population.units
                if unit.raw_length < config.window_length
            )
        for name, population in self.references.items():
            rng = np.random.default_rng(config.random_state)
            intervals: list[tuple[int, int, int]] = []
            for sequence, size in enumerate(np.diff(population.cache.offsets)):
                candidates = range(0, int(size) - row_length + 1, config.reference_stride)
                cap = config.reference_windows_per_trajectory
                starts: Iterable[int]
                if cap is not None and len(candidates) > cap:
                    starts = np.sort(rng.choice(len(candidates), size=cap, replace=False))
                    starts = starts * config.reference_stride
                else:
                    starts = candidates
                intervals.extend(
                    (sequence, int(start), int(start) + row_length) for start in starts
                )
            bank = population.cache.aggregate(intervals)
            units = [population.unit(interval, config.window_length) for interval in intervals]
            banks[name] = bank, units
            reference_rows.extend(
                {**unit.record(), **common, "reference_group": name} for unit in units
            )
        tables = []
        for batch in self.query.cache.iter_windows(
            row_length, stride=config.query_stride, batch_size=config.query_batch_size
        ):
            units = [
                self.query.unit(interval, config.window_length) for interval in batch.intervals
            ]
            base = pd.DataFrame([{**unit.record(), **common} for unit in units])
            for metric in config.metrics:
                if metric == "outside-mass":
                    table = base.copy()
                    table["metric"] = metric
                    table["reference_group"] = "__basis__"
                    table["aggregation"] = "partition_mean"
                    table["score"] = batch.outside_mass(atol=1e-10).mean
                    table["eligible_reference_count"] = 0
                    table["selected_count"] = 0
                    tables.append(table)
                    continue
                parameters = {"atol": 1e-10} if metric in {"js", "kl"} else {}
                if metric == "kl":
                    assert config.epsilon is not None
                    parameters["epsilon"] = config.epsilon
                for name, (bank, reference_units) in banks.items():

                    def eligibility(
                        q_indices: np.ndarray,
                        r_indices: np.ndarray,
                        query_units: list[AnalysisUnit] = units,
                        bank_units: list[AnalysisUnit] = reference_units,
                    ) -> np.ndarray:
                        return eligible_references(
                            [query_units[i] for i in q_indices],
                            [bank_units[i] for i in r_indices],
                            exclude_same_unit=config.exclude_same_unit,
                            exclude_same_trajectory=config.exclude_same_trajectory,
                            max_overlap=config.max_overlap,
                        )

                    result = reduce_references(
                        batch,
                        bank,
                        metric=PYIDK_METRICS[metric],
                        metric_parameters=parameters,
                        aggregation=config.aggregation,
                        k=config.k,
                        eligibility=eligibility,
                        query_block_size=config.query_batch_size,
                        reference_block_size=config.reference_block_size,
                    )
                    table = base.copy()
                    table["metric"] = metric
                    table["reference_group"] = name
                    table["aggregation"] = config.aggregation
                    table["score"] = result.scores
                    table["eligible_reference_count"] = result.eligible_reference_count
                    table["selected_count"] = result.selected_count
                    tables.append(table)
        scores = (
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=SCORE_COLUMNS)
        )
        references = (
            pd.DataFrame(reference_rows)
            if reference_rows
            else pd.DataFrame(columns=REFERENCE_COLUMNS)
        )
        return WindowAnalyticsResult(
            scores,
            summarize_window_scores(scores),
            references,
            pd.DataFrame(
                skipped, columns=None if skipped else ["population", "trajectory_id", "reason"]
            ),
            {
                **metadata,
                "analysis_id": analysis_id,
                "software_versions": software_versions(),
                "summary": "Query-score mean/median/IQR by local end_step and source_dataset",
                "intervals": "Raw transitions [start_step, end_step); available at end_step",
                "reference_weighting": "Equal eligible windows; top_k selects over the whole bank",
            },
        )
