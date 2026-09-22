"""Analysis-unit provenance; raw data is optional and never required by analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cartpole_idk.storage.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class AnalysisUnit:
    """One whole trajectory or contiguous window represented by a single embedding.

    The half-open [start_step, end_step) interval is local to trajectory_id.
    Metadata retains source provenance; trajectory holds the raw segment during
    fitting/embedding and is absent when loaded for analytics.
    """

    unit_id: str
    trajectory_id: str
    start_step: int
    end_step: int
    metadata: dict[str, Any]
    trajectory: Trajectory | None = field(default=None, repr=False, compare=False)

    @property
    def raw_length(self) -> int:
        """Number of transitions in this unit, excluding the final boundary observation."""
        return self.end_step - self.start_step

    @property
    def source_key(self) -> tuple[str, str]:
        """Original dataset and trajectory, including provenance of prepared segments."""
        return (
            self.metadata.get("source_dataset", ""),
            self.metadata.get("source_trajectory_id", self.trajectory_id),
        )

    def record(self) -> dict[str, Any]:
        """Flatten provenance and interval fields into a row for tables and reports."""
        return {
            **self.metadata,
            "unit_id": self.unit_id,
            "trajectory_id": self.trajectory_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "raw_length": self.raw_length,
        }
