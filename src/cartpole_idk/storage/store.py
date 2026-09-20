from __future__ import annotations

from pathlib import Path

import pandas as pd

from cartpole_idk.storage.trajectory import Trajectory


class TrajectoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.trajectory_dir = self.root / "trajectories"
        self.manifest_path = self.root / "manifest.parquet"

    def initialize(self) -> None:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def add(self, trajectory: Trajectory) -> None:
        self.initialize()
        trajectory.save(self.trajectory_dir / f"{trajectory.trajectory_id}.npz")
        row = {
            "trajectory_id": trajectory.trajectory_id,
            "length": trajectory.length,
            "return": trajectory.episode_return,
            **trajectory.metadata,
        }
        df = pd.read_parquet(self.manifest_path) if self.manifest_path.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(self.manifest_path, index=False)

    def manifest(self) -> pd.DataFrame:
        return (
            pd.read_parquet(self.manifest_path) if self.manifest_path.exists() else pd.DataFrame()
        )

    def get(self, trajectory_id: str) -> Trajectory:
        path = self.trajectory_dir / f"{trajectory_id}.npz"
        if not path.exists():
            raise KeyError(f"Unknown trajectory_id: {trajectory_id}")
        return Trajectory.load(path)

    def query(
        self,
        *,
        checkpoint_id=None,
        perturbation_type=None,
        min_return=None,
        max_return=None,
        min_length=None,
        max_length=None,
    ) -> pd.DataFrame:
        df = self.manifest()
        if df.empty:
            return df
        if checkpoint_id is not None:
            df = df[df["checkpoint_id"] == checkpoint_id]
        if perturbation_type is not None:
            df = df[df["perturbation_type"] == perturbation_type]
        if min_return is not None:
            df = df[df["return"] >= min_return]
        if max_return is not None:
            df = df[df["return"] <= max_return]
        if min_length is not None:
            df = df[df["length"] >= min_length]
        if max_length is not None:
            df = df[df["length"] <= max_length]
        return df.reset_index(drop=True)
