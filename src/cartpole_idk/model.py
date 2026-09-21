"""Typed models for persisted experiment reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ReturnStatistics(BaseModel):
    """Population standard deviation and linearly interpolated quartiles."""

    mean: float
    std_dev: float
    min: float
    max: float
    q1: float
    median: float
    q3: float

    @staticmethod
    def from_returns(returns: list[float]) -> ReturnStatistics | None:
        if not returns:
            return None
        values = np.asarray(returns, dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        return ReturnStatistics(
            mean=float(values.mean()),
            std_dev=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            q1=float(q1),
            median=float(median),
            q3=float(q3),
        )


class GeneratedTrajectory(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trajectory_id: str
    episode_return: float = Field(validation_alias="return", serialization_alias="return")
    length: int
    perturbation_onset: int | None = None


class GenerationReport(BaseModel):
    checkpoint: str
    episodes: int
    seed: int
    perturbation_type: str
    generated: list[GeneratedTrajectory]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def return_statistics(self) -> ReturnStatistics | None:
        return ReturnStatistics.from_returns([row.episode_return for row in self.generated])

    @staticmethod
    def from_file(path: str | Path) -> GenerationReport:
        """Load a JSON report, including reports written before statistics existed."""
        return GenerationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2, by_alias=True), encoding="utf-8")


# Pipeline contracts contain configuration and provenance, never numeric array payloads.


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class IDKConfig(PipelineModel):
    representation: Literal[
        "state", "transition", "state_action", "state_action_next_state", "window"
    ] = "state"
    observation_source: Literal["true", "agent"] = "true"
    action_source: Literal["commanded", "executed"] = "executed"
    window_length: int = Field(default=25, ge=1)
    psi: int = Field(default=32, ge=2)
    t: int = Field(default=200, ge=1)
    random_state: int | None = Field(default=42, ge=0)
    scaler: Literal["standard"] = "standard"

    def feature_width(self, observation_width: int) -> int:
        if self.representation == "transition":
            return 2 * observation_width
        if self.representation == "state_action":
            return observation_width + 2
        if self.representation == "state_action_next_state":
            return 2 * observation_width + 2
        if self.representation == "window":
            return self.window_length * observation_width
        return observation_width


class WindowConfig(PipelineModel):
    mode: Literal["whole", "window"] = "whole"
    window_length: int = Field(default=25, ge=1)
    stride: int = Field(default=1, ge=1)


class DatasetSelection(PipelineModel):
    dataset: Path
    trajectory_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_ids(self) -> Self:
        if any(not tid for tid in self.trajectory_ids) or len(set(self.trajectory_ids)) != len(
            self.trajectory_ids
        ):
            raise ValueError("Trajectory IDs must be nonempty and distinct")
        return self


class FitConfig(DatasetSelection):
    idk: IDKConfig = Field(default_factory=IDKConfig)
    unit: WindowConfig = Field(default_factory=WindowConfig)


class EmbedConfig(DatasetSelection):
    fit_artifact: Path
    fit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit: FitConfig
    unit: WindowConfig = Field(default_factory=WindowConfig)


class MetricConfig(PipelineModel):
    metric: Literal["idk", "idk-distance", "cosine", "js", "kl"] = "idk"
    epsilon: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def kl_smoothing(self) -> Self:
        if self.metric == "kl" and self.epsilon is None:
            raise ValueError("--metric kl requires --epsilon > 0")
        if self.metric != "kl" and self.epsilon is not None:
            raise ValueError("epsilon is only applicable to KL")
        return self


class NeighborConfig(PipelineModel):
    k: int = Field(default=5, ge=1)
    include_self: bool = False
    exclude_same_trajectory: bool = False
    max_overlap: float | None = Field(default=None, ge=0, le=1)


class ClusterConfig(PipelineModel):
    algorithm: Literal["hdbscan", "dbscan", "spectral", "dpgmm"] = "hdbscan"
    distance: Literal["idk-distance", "js"] = "idk-distance"
    eps: float = Field(default=0.1, gt=0)
    min_samples: int | None = Field(default=None, ge=1)
    min_cluster_size: int = Field(default=5, ge=2)
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    n_clusters: int = Field(default=2, ge=1)
    random_state: int = Field(default=42, ge=0)
    svd_components: int = Field(default=10, ge=1)
    n_components: int = Field(default=10, ge=1)
    max_iter: int = Field(default=500, ge=1)
    weight_concentration_prior: float | None = Field(default=None, gt=0)
    categorical: tuple[str, ...] = ("perturbation_type", "checkpoint_id")

    @model_validator(mode="after")
    def algorithm_options(self) -> Self:
        if self.algorithm in {"spectral", "dpgmm"} and self.distance != "idk-distance":
            raise ValueError("distance only applies to density clustering")
        return self


class PopulationConfig(PipelineModel):
    bandwidth: float | None = Field(default=None, gt=0)
    estimator: Literal["biased", "unbiased"] = "biased"
    permutations: int = Field(default=0, ge=0)
    random_state: int = Field(default=42, ge=0)


class AnalysisConfig(PipelineModel):
    command: Literal["pairwise", "neighbors", "rolling", "cluster", "population"]
    embeddings: Path
    output: Path
    reference: Path | None = None
    nominal_reference: Path | None = None
    failure_reference: Path | None = None
    group_b: Path | None = None
    query_id: str | None = None
    max_pairs: int = Field(default=4_000_000, ge=1)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    neighbors: NeighborConfig = Field(default_factory=NeighborConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)

    @model_validator(mode="after")
    def population_roles(self) -> Self:
        if (self.nominal_reference is None) != (self.failure_reference is None):
            raise ValueError("Provide both nominal and failure reference artifacts")
        if self.reference is not None and self.nominal_reference is not None:
            raise ValueError("Use reference or nominal/failure references, not both")
        if self.command == "population" and self.group_b is None:
            raise ValueError("population requires --group-b")
        if self.command == "rolling" and not self.query_id:
            raise ValueError("rolling requires --query-id")
        if self.command not in {"neighbors", "rolling"} and self.nominal_reference is not None:
            raise ValueError("Nominal/failure references require neighbors or rolling")
        if self.command not in {"pairwise", "neighbors", "rolling"} and self.reference is not None:
            raise ValueError("Reference artifact is not applicable to this command")
        if self.command != "population" and self.group_b is not None:
            raise ValueError("group_b requires population analysis")
        if self.command != "rolling" and self.query_id is not None:
            raise ValueError("query_id requires rolling analysis")
        return self


class ArtifactMetadata(PipelineModel):
    format_version: Literal[1] = 1
    kind: Literal["fit", "embedding"]
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]
    software: dict[str, str]
    n_units: int = Field(ge=1)
    n_features: int = Field(ge=1)
    observation_width: int = Field(ge=1)
    n_fit_samples: int = Field(ge=2)
    skipped: tuple[dict[str, object], ...] = ()
    parent_fit_id: str | None = None


class UnitRecord(PipelineModel):
    unit_id: str
    trajectory_id: str
    start_step: int = Field(ge=0)
    end_step: int = Field(gt=0)
    mode: Literal["whole", "window"]
    metadata: dict[str, object]

    @model_validator(mode="after")
    def interval(self) -> Self:
        if self.end_step <= self.start_step:
            raise ValueError("Unit end must follow its start")
        return self


class AnalysisInput(PipelineModel):
    path: Path
    artifact_id: str
    fit_id: str


class AnalysisMetadata(PipelineModel):
    format_version: Literal[1] = 1
    inputs: dict[str, AnalysisInput]
    software: dict[str, str]
    unit_roles: dict[str, tuple[str, ...]]
