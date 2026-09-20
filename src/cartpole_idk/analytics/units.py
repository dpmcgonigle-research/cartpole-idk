"""Episode-bounded analysis units. Step intervals are half-open transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from cartpole_idk.storage import Trajectory


@dataclass(frozen=True, slots=True)
class AnalysisUnit:
    unit_id: str
    trajectory_id: str
    start_step: int
    end_step: int
    metadata: dict[str, Any]
    trajectory: Trajectory = field(repr=False, compare=False)

    @property
    def raw_length(self) -> int:
        return self.end_step - self.start_step

    @property
    def source_key(self) -> tuple[str, str]:
        """Original dataset and trajectory, including provenance of prepared segments."""
        return (
            self.metadata.get("source_dataset", ""),
            self.metadata.get("source_trajectory_id", self.trajectory_id),
        )

    def record(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "unit_id": self.unit_id,
            "trajectory_id": self.trajectory_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "raw_length": self.raw_length,
        }


@dataclass(slots=True)
class UnitCollection:
    units: list[AnalysisUnit]
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([unit.record() for unit in self.units])


def build_units(
    trajectories: Sequence[Trajectory],
    *,
    mode: Literal["whole", "window"] = "whole",
    window_length: int = 25,
    stride: int = 1,
) -> UnitCollection:
    """Retain complete windows only; never pad or cross an episode boundary.

    An L-transition segment includes L+1 observations. Episode termination is only
    known when a stored terminal/truncation flag is present. Prepared prefixes are
    therefore censored, even when the original episode length is in provenance.
    """
    if mode not in {"whole", "window"} or window_length < 1 or stride < 1:
        raise ValueError("Require whole/window mode and positive window_length and stride")
    if len({t.trajectory_id for t in trajectories}) != len(trajectories):
        raise ValueError("Trajectory IDs must be unique within an analysis population")
    result = UnitCollection([])
    for traj in trajectories:
        if traj.length == 0 or (mode == "window" and traj.length < window_length):
            result.skipped.append({"trajectory_id": traj.trajectory_id, "reason": "too_short"})
            continue
        for name in ("true_observations", "agent_observations"):
            if len(getattr(traj, name)) != traj.length + 1:
                raise ValueError(f"{traj.trajectory_id}: {name} must have T+1 observations")
        for name in ("commanded_actions", "rewards", "terminated", "truncated"):
            if len(getattr(traj, name)) != traj.length:
                raise ValueError(f"{traj.trajectory_id}: {name} must have T timesteps")
        if (traj.terminated[:-1] | traj.truncated[:-1]).any():
            raise ValueError(f"{traj.trajectory_id}: contains an internal episode boundary")
        length = traj.length if mode == "whole" else window_length
        starts = [0] if mode == "whole" else range(0, traj.length - length + 1, stride)
        for start in starts:
            end = start + length
            offset = int(traj.metadata.get("segment_start", 0))
            onset = traj.metadata.get("perturbation_onset")
            terminal = bool(traj.terminated[-1] or traj.truncated[-1])
            metadata = {
                **traj.metadata,
                "episode_length": traj.metadata.get("source_length", traj.length),
                "episode_return": traj.metadata.get("source_return", traj.episode_return),
                "stored_length": traj.length,
                "segment_return": float(traj.rewards[start:end].sum()),
                "source_start_step": offset + start,
                "source_end_step": offset + end,
                "episode_end_step": offset + traj.length if terminal else None,
                "time_until_termination": traj.length - end if terminal else None,
                "time_until_failure": traj.length - end if traj.terminated[-1] else None,
                "time_since_perturbation": end - onset if onset is not None else None,
                "terminated": bool(traj.terminated[-1]),
                "truncated": bool(traj.truncated[-1]),
            }
            uid = f"{traj.trajectory_id}:{start}:{end}"
            segment = Trajectory(
                uid,
                traj.true_observations[start : end + 1],
                traj.agent_observations[start : end + 1],
                traj.commanded_actions[start:end],
                traj.executed_actions[start:end],
                traj.rewards[start:end],
                traj.terminated[start:end],
                traj.truncated[start:end],
                dict(metadata),
            )
            result.units.append(
                AnalysisUnit(uid, traj.trajectory_id, start, end, metadata, segment)
            )
    return result
