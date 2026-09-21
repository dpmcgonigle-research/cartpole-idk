import numpy as np
import pytest

from cartpole_idk.storage import Trajectory


@pytest.fixture
def trajectory():
    true_obs = np.array(
        [
            [0.0, 0.0, 0.01, 0.0],
            [0.01, 0.1, 0.02, 0.2],
            [0.02, 0.2, 0.03, 0.3],
            [0.03, 0.3, 0.04, 0.4],
        ],
        dtype=np.float32,
    )
    return Trajectory(
        trajectory_id="traj_test",
        true_observations=true_obs,
        agent_observations=true_obs.copy(),
        commanded_actions=np.array([0, 1, 0], dtype=np.int8),
        executed_actions=np.array([0, 1, 0], dtype=np.int8),
        rewards=np.ones(3, dtype=np.float32),
        terminated=np.array([False, False, True]),
        truncated=np.array([False, False, False]),
        metadata={
            "checkpoint_id": "step_100",
            "perturbation_type": "none",
            "perturbation_onset": None,
        },
    )


@pytest.fixture
def pipeline_dataset(tmp_path, trajectory):
    from dataclasses import replace

    from cartpole_idk.storage import TrajectoryStore

    store = TrajectoryStore(tmp_path / "dataset")
    for i in range(8):
        store.add(
            replace(
                trajectory,
                trajectory_id=f"traj_{i}",
                true_observations=trajectory.true_observations + i * 0.01,
                metadata={
                    "perturbation_type": "none" if i < 4 else "flip",
                    "perturbation": {"onset": 1, "parameters": {"bias": 0.1}},
                },
            )
        )
    return store


@pytest.fixture
def saved_fit(tmp_path, pipeline_dataset):
    from cartpole_idk.artifacts import FitArtifact
    from cartpole_idk.idk.pipeline import fit_dataset
    from cartpole_idk.model import FitConfig, IDKConfig

    config = FitConfig(
        dataset=pipeline_dataset.root,
        trajectory_ids=("traj_0", "traj_1", "traj_4", "traj_5"),
        idk=IDKConfig(representation="transition", psi=2, t=4),
    )
    path = tmp_path / "fit"
    fit_dataset(config).save(path)
    return FitArtifact.load(path)


@pytest.fixture
def embedding_artifacts(tmp_path, pipeline_dataset, saved_fit):
    from cartpole_idk.idk.pipeline import embed_dataset
    from cartpole_idk.model import EmbedConfig, WindowConfig

    paths = {}
    for name, ids in {
        "test": [2, 3, 6, 7],
        "nominal": [0, 1],
        "failure": [4, 5],
        "a": [2, 3],
        "b": [6, 7],
        "whole": [2, 3, 6, 7],
    }.items():
        config = EmbedConfig(
            dataset=pipeline_dataset.root,
            trajectory_ids=tuple(f"traj_{i}" for i in ids),
            fit_artifact=tmp_path / "fit",
            fit_id=saved_fit.fit_id,
            fit=saved_fit.config,
            unit=WindowConfig(mode="whole" if name == "whole" else "window", window_length=2),
        )
        paths[name] = tmp_path / f"embeddings_{name}"
        embed_dataset(saved_fit, config).save(paths[name])
    return paths
