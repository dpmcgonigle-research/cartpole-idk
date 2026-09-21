import json
import shutil

import numpy as np
import pytest
from pyidk import IsolationDistributionalKernel, IsolationKernel, Standardizer

from cartpole_idk.artifacts import EmbeddingArtifact, FitArtifact
from cartpole_idk.artifacts.common import checksums, content_id
from cartpole_idk.idk.embedding import represented_units
from cartpole_idk.idk.pipeline import embed_dataset, fit_dataset, selected_units
from cartpole_idk.model import (
    AnalysisConfig,
    ClusterConfig,
    EmbedConfig,
    FitConfig,
    IDKConfig,
    MetricConfig,
    NeighborConfig,
    PopulationConfig,
    WindowConfig,
)


def refresh_checksums(root):
    path = root / "metadata.json"
    meta = json.loads(path.read_text())
    meta["files"] = checksums(root, tuple(meta["files"]))
    meta["artifact_id"] = content_id(meta["files"])
    path.write_text(json.dumps(meta))


def test_fit_roundtrip_and_identical_transforms(saved_fit, tmp_path):
    original = fit_dataset(saved_fit.config)
    original.save(tmp_path / "again")
    restored = FitArtifact.load(tmp_path / "again")
    units = selected_units(original.config, original.config.unit)
    _, batch = represented_units(units, original.config.idk)
    np.testing.assert_array_equal(original.scaler.mean_, restored.scaler.mean_)
    np.testing.assert_array_equal(original.scaler.scale_, restored.scaler.scale_)
    scaled = batch.with_values(original.scaler.transform(batch.values))
    np.testing.assert_array_equal(
        original.model.transform(scaled).toarray(), restored.model.transform(scaled).toarray()
    )
    assert original.fit_id == restored.fit_id == saved_fit.fit_id
    shutil.copytree(tmp_path / "fit", tmp_path / "copy")
    assert FitArtifact.load(tmp_path / "copy").fit_id == saved_fit.fit_id
    assert set(p.name for p in (tmp_path / "fit").iterdir()) == {
        "config.json",
        "metadata.json",
        "scaler.npz",
        "basis.npz",
        "fit_manifest.parquet",
    }


def test_embed_roundtrip_and_never_refits(saved_fit, pipeline_dataset, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Embedding must never fit")

    for cls in [IsolationKernel, IsolationDistributionalKernel, Standardizer]:
        monkeypatch.setattr(cls, "fit", forbidden)
    fit = FitArtifact.load(tmp_path / "fit")
    config = EmbedConfig(
        dataset=pipeline_dataset.root,
        trajectory_ids=("traj_2", "traj_3"),
        fit_artifact=tmp_path / "fit",
        fit_id=fit.fit_id,
        fit=fit.config,
        unit=WindowConfig(mode="window", window_length=2),
    )
    artifact = embed_dataset(fit, config)
    artifact.save(tmp_path / "embedded")
    loaded = EmbeddingArtifact.load(tmp_path / "embedded")
    np.testing.assert_array_equal(
        artifact.embeddings.values.toarray(), loaded.embeddings.values.toarray()
    )
    assert [u.record() for u in loaded.embeddings.units] == [
        u.record() for u in artifact.embeddings.units
    ]
    assert all(u.trajectory is None for u in loaded.embeddings.units)
    assert loaded.config == config
    assert loaded.metadata.parent_fit_id == saved_fit.fit_id
    assert set(p.name for p in (tmp_path / "embedded").iterdir()) == {
        "config.json",
        "units.parquet",
        "embeddings.npz",
        "metadata.json",
    }


@pytest.mark.parametrize(
    "model,kwargs",
    [
        (IDKConfig, {"psi": 1}),
        (IDKConfig, {"t": 0}),
        (IDKConfig, {"random_state": -1}),
        (IDKConfig, {"representation": "bad"}),
        (WindowConfig, {"window_length": 0}),
        (WindowConfig, {"stride": 0}),
        (MetricConfig, {"metric": "kl"}),
        (MetricConfig, {"metric": "kl", "epsilon": 0}),
        (NeighborConfig, {"max_overlap": 1.1}),
        (NeighborConfig, {"k": 0}),
        (PopulationConfig, {"bandwidth": float("nan")}),
        (ClusterConfig, {"min_cluster_size": 1}),
        (ClusterConfig, {"algorithm": "spectral", "distance": "js"}),
    ],
)
def test_config_validation(model, kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)


def test_models_serialize_and_are_frozen(saved_fit, tmp_path):
    assert FitConfig.model_validate_json(saved_fit.config.model_dump_json()) == saved_fit.config
    with pytest.raises(ValueError):
        saved_fit.config.idk.psi = 4
    config = AnalysisConfig(
        command="pairwise", embeddings=tmp_path / "embed", output=tmp_path / "out"
    )
    assert AnalysisConfig.model_validate_json(config.model_dump_json()) == config
    assert "properties" in AnalysisConfig.model_json_schema()
    with pytest.raises(ValueError):
        AnalysisConfig(command="pairwise", embeddings=tmp_path, output=tmp_path, psi=32)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("version", "format_version"),
        ("missing", "Missing required"),
        ("checksum", "checksum"),
        ("scaler", "Scaler dimensionality"),
        ("radii", "radii"),
        ("indices", "sample indices"),
        ("representation", "dimensionality"),
    ],
)
def test_malformed_fit_artifacts(saved_fit, tmp_path, mutation, error):
    root = tmp_path / "fit"
    if mutation == "version":
        p = root / "metadata.json"
        data = json.loads(p.read_text())
        data["format_version"] = 999
        p.write_text(json.dumps(data))
    elif mutation == "missing":
        (root / "scaler.npz").unlink()
    elif mutation == "checksum":
        (root / "config.json").write_text("{}")
    elif mutation == "scaler":
        np.savez_compressed(root / "scaler.npz", mean=np.ones(2), scale=np.ones(2))
        refresh_checksums(root)
    elif mutation in {"radii", "indices"}:
        with np.load(root / "basis.npz") as z:
            arrays = {key: z[key] for key in z.files}
        if mutation == "radii":
            arrays["radii"][:] = -1
        else:
            arrays["sample_indices"][:] = 9999
        np.savez_compressed(root / "basis.npz", **arrays)
        refresh_checksums(root)
    else:
        p = root / "config.json"
        data = json.loads(p.read_text())
        data["idk"]["representation"] = "state"
        p.write_text(json.dumps(data))
        refresh_checksums(root)
    with pytest.raises(ValueError, match=error):
        FitArtifact.load(root)


def test_malformed_embeddings_and_incompatible_fit(embedding_artifacts, saved_fit, tmp_path):
    from scipy.sparse import csr_matrix, save_npz

    root = embedding_artifacts["test"]
    good = EmbeddingArtifact.load(root)
    save_npz(root / "embeddings.npz", csr_matrix((len(good.embeddings.units), 7)))
    refresh_checksums(root)
    with pytest.raises(ValueError, match="shape"):
        EmbeddingArtifact.load(root)
    other_config = FitConfig.model_validate(
        {
            **saved_fit.config.model_dump(),
            "idk": {**saved_fit.config.idk.model_dump(), "random_state": 21},
        }
    )
    other = fit_dataset(other_config)
    other.save(tmp_path / "other_fit")
    other_embed_config = EmbedConfig.model_validate(
        {**good.config.model_dump(), "fit": other.config, "fit_id": other.fit_id}
    )
    different = embed_dataset(other, other_embed_config)
    with pytest.raises(ValueError, match="same fitted basis"):
        good.compatible_with(different)


@pytest.mark.parametrize(
    "representation", ["state", "transition", "state_action", "state_action_next_state", "window"]
)
def test_all_representations_persist(pipeline_dataset, tmp_path, representation):
    config = FitConfig(
        dataset=pipeline_dataset.root,
        trajectory_ids=("traj_0", "traj_1"),
        idk=IDKConfig(
            representation=representation,
            window_length=2,
            observation_source="agent",
            action_source="commanded",
            psi=2,
            t=3,
        ),
    )
    original = fit_dataset(config)
    original.save(tmp_path / "fit")
    loaded = FitArtifact.load(tmp_path / "fit")
    embed_config = EmbedConfig(
        dataset=pipeline_dataset.root,
        trajectory_ids=("traj_2",),
        fit_artifact=tmp_path / "fit",
        fit_id=loaded.fit_id,
        fit=loaded.config,
    )
    first, second = embed_dataset(original, embed_config), embed_dataset(loaded, embed_config)
    np.testing.assert_array_equal(
        first.embeddings.values.toarray(), second.embeddings.values.toarray()
    )


def test_missing_embedding_version_and_row_order(embedding_artifacts):
    import pandas as pd

    root = embedding_artifacts["test"]
    table = pd.read_parquet(root / "units.parquet")
    table.loc[0, "embedding_row"] = 999
    table.to_parquet(root / "units.parquet", index=False)
    refresh_checksums(root)
    with pytest.raises(ValueError, match="ordering"):
        EmbeddingArtifact.load(root)
    (root / "embeddings.npz").unlink()
    with pytest.raises(ValueError, match="Missing required"):
        EmbeddingArtifact.load(root)


def test_sparse_occupancy_validation(embedding_artifacts):
    from scipy.sparse import save_npz

    root = embedding_artifacts["test"]
    artifact = EmbeddingArtifact.load(root)
    artifact.embeddings.values.data[:] = 2
    save_npz(root / "embeddings.npz", artifact.embeddings.values)
    refresh_checksums(root)
    with pytest.raises(ValueError, match="occupancy"):
        EmbeddingArtifact.load(root)


def test_failed_artifact_write_does_not_publish(saved_fit, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="interrupted"):
        saved_fit.save(tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_fit_uses_selected_observation_dimension(tmp_path, trajectory):
    from dataclasses import replace

    from cartpole_idk.storage import TrajectoryStore

    store = TrajectoryStore(tmp_path / "different_widths")
    store.add(replace(trajectory, agent_observations=trajectory.agent_observations[:, :2]))
    config = FitConfig(
        dataset=store.root,
        trajectory_ids=(trajectory.trajectory_id,),
        idk=IDKConfig(observation_source="agent", psi=2, t=2),
    )
    artifact = fit_dataset(config)
    assert artifact.observation_width == 2
    artifact.save(tmp_path / "agent_fit")
    assert FitArtifact.load(tmp_path / "agent_fit").scaler.mean_.shape == (2,)
