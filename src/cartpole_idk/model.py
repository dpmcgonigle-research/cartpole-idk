"""Typed models for persisted experiment reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field


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
