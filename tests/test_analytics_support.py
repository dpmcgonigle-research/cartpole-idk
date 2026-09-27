import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pyidk import IsolationDistributionalKernel, IsolationKernel, Standardizer
from scipy.sparse import csr_matrix

from cartpole_idk.analytics.support import (
    SUPPORT_METRICS,
    partition_outside_mass,
    summarize_support,
    support_dataframe,
    support_group_summary,
    support_summary,
)
from cartpole_idk.artifacts import EmbeddingArtifact, FitArtifact
from cartpole_idk.cli.analyze import main
from cartpole_idk.model import EmbedConfig, FitConfig, IDKConfig, WindowConfig
from cartpole_idk.storage import TrajectoryStore
from cartpole_idk.storage.embeddings import EmbeddingSet
from cartpole_idk.storage.units import AnalysisUnit


@pytest.mark.parametrize("block, expected", [([0.4, 0.6], 0), ([0.2, 0.5], 0.3), ([0, 0], 1)])
def test_partition_occupancy(block, expected):
    values = csr_matrix([block])
    original = values.copy()
    outside = partition_outside_mass(values, t=1, psi=2)
    assert outside.shape == (1, 1)
    assert outside[0, 0] == pytest.approx(expected)
    assert (values != original).nnz == 0


def test_multi_partition_summary():
    values = csr_matrix([[0.4, 0.6, 0.2, 0.5, 0, 0]])
    outside = partition_outside_mass(values, t=3, psi=2)
    np.testing.assert_allclose(outside, [[0, 0.3, 1]])
    row = summarize_support(outside).iloc[0]
    assert row.outside_mass_mean == pytest.approx(1.3 / 3)
    assert row.outside_mass_median == pytest.approx(0.3)
    assert row.outside_mass_min == 0
    assert row.outside_mass_max == 1
    assert row.outside_mass_std == pytest.approx(np.std([0, 0.3, 1], ddof=0))
    assert row.partitions_with_any_outside == 2
    assert row.support_rate_mean == pytest.approx(1 - 1.3 / 3)


def test_only_partition_sums_are_densified(monkeypatch):
    original = csr_matrix.toarray
    seen = []

    def checked(self, *args, **kwargs):
        seen.append(self.shape)
        assert self.shape == (7, 3)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", checked)
    result = partition_outside_mass(csr_matrix((7, 300)), t=3, psi=100)
    np.testing.assert_array_equal(result, np.ones((7, 3)))
    assert seen == [(7, 3)]


def test_width_mismatch():
    with pytest.raises(ValueError, match=r"Embedding width 5 != t \* psi \(6\)"):
        partition_outside_mass(csr_matrix((2, 5)), t=3, psi=2)


@pytest.mark.parametrize("data", [[np.nan, 0], [np.inf, 0], [-0.1, 0.5], [1.1, 0], [0.8, 0.4]])
def test_invalid_occupancies_rejected(data):
    with pytest.raises(ValueError):
        partition_outside_mass(csr_matrix([data]), t=1, psi=2)


def test_roundoff_clipped_but_small_missing_mass_retained():
    values = csr_matrix([[0.5, 0.5 + 1e-12], [-1e-12, 0.7], [0.5, 0.5 - 1e-12]])
    outside = partition_outside_mass(values, t=1, psi=2)
    assert outside[0, 0] == 0
    assert outside[1, 0] == pytest.approx(0.3)
    assert outside[2, 0] > 0
    assert summarize_support(outside).partitions_with_any_outside.tolist() == [0, 1, 1]


@pytest.mark.parametrize("t,psi", [(0, 2), (2, 0), (1.5, 2), (True, 2)])
def test_invalid_partition_dimensions(t, psi):
    with pytest.raises(ValueError, match="positive integers"):
        partition_outside_mass(csr_matrix((2, 4)), t=t, psi=psi)


@pytest.fixture
def synthetic_support_artifact(tmp_path):
    ids = ("zeta", "alpha", "beta")
    unit_config = WindowConfig(mode="window", window_length=25, stride=1)
    fit_config = FitConfig(
        dataset=tmp_path / "absent_training_data",
        trajectory_ids=ids,
        idk=IDKConfig(t=3, psi=2),
    )
    config = EmbedConfig(
        dataset=tmp_path / "absent_query_data",
        trajectory_ids=ids,
        fit_artifact=tmp_path / "absent_fit",
        fit_id="a" * 64,
        fit=fit_config,
        unit=unit_config,
    )
    units = [
        AnalysisUnit(
            f"unit_{tid}",
            tid,
            i,
            i + 25,
            {
                "source_trajectory_id": f"original_{tid}",
                "source_dataset": "source_dataset",
                "source_start_step": 10 + i,
                "source_end_step": 35 + i,
                "stored_length": 30,
                "group": "A" if i < 2 else "B",
                "nested": {"value": i},
            },
        )
        for i, tid in enumerate(ids)
    ]
    values = csr_matrix([[0.4, 0.6, 0.2, 0.5, 0, 0], [0, 0, 0, 0, 0, 0], [0.5] * 6])
    embedded = EmbeddingSet(units, values, config.fit_id, 3, 2)
    artifact = EmbeddingArtifact(config, embedded, observation_width=4, n_fit_samples=10)
    path = tmp_path / "embeddings"
    artifact.save(path)
    return path


def test_metadata_and_unit_order(synthetic_support_artifact):
    embedded = EmbeddingArtifact.load(synthetic_support_artifact).embeddings
    outside = partition_outside_mass(embedded.values, t=embedded.t, psi=embedded.psi)
    frame = support_dataframe(embedded, outside)
    assert frame.unit_id.tolist() == ["unit_zeta", "unit_alpha", "unit_beta"]
    assert frame.source_trajectory_id.tolist() == [
        "original_zeta",
        "original_alpha",
        "original_beta",
    ]
    assert frame.embedding_row.tolist() == [0, 1, 2]
    assert frame.start_step.tolist() == [0, 1, 2]
    assert frame.end_step.tolist() == [25, 26, 27]
    assert frame.source_start_step.tolist() == [10, 11, 12]
    assert frame.nested.tolist() == [{"value": 0}, {"value": 1}, {"value": 2}]
    np.testing.assert_allclose(frame.outside_mass_mean, [1.3 / 3, 1, 0])
    with pytest.raises(ValueError, match="shape"):
        support_dataframe(embedded, outside[:2])
    stats = support_summary(frame, t=3, psi=2)
    assert stats.n_units == 3
    assert stats.statistics["outside_mass_mean"].mean == pytest.approx((1.3 / 3 + 1) / 3)


def test_group_summaries_and_missing_groups():
    frame = summarize_support(np.array([[0, 0.5], [1, 1], [0, 0]]))
    frame["source"] = ["A", "A", None]
    grouped = support_group_summary(frame, "source")
    means = grouped[grouped.metric == "outside_mass_mean"]
    assert means.n_units.tolist() == [2, 1]
    np.testing.assert_allclose(means["mean"], [0.625, 0])
    assert pd.isna(means.iloc[1].group_value)
    with pytest.raises(ValueError, match="Unknown group-by"):
        support_group_summary(frame, "missing")
    frame["nested"] = [{"a": 1}] * 3
    with pytest.raises(ValueError, match="scalar"):
        support_group_summary(frame, "nested")


@pytest.mark.parametrize("grouped", [False, True])
def test_support_cli(synthetic_support_artifact, tmp_path, monkeypatch, grouped):
    def forbidden(*args, **kwargs):
        pytest.fail("Support must not fit, transform, or load raw data")

    for cls in (IsolationKernel, IsolationDistributionalKernel, Standardizer):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "transform", forbidden)
    monkeypatch.setattr(TrajectoryStore, "get", forbidden)
    monkeypatch.setattr(FitArtifact, "load", forbidden)
    output = tmp_path / "support"
    args = ["support", str(synthetic_support_artifact), "--output", str(output)]
    if grouped:
        args += ["--group-by", "group"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(output / "support.parquet")
    assert table.unit_id.tolist() == ["unit_zeta", "unit_alpha", "unit_beta"]
    assert set(SUPPORT_METRICS) <= set(table)
    assert table["nested"].map(json.loads).tolist() == [{"value": 0}, {"value": 1}, {"value": 2}]
    summary = json.loads((output / "support_summary.json").read_text())
    assert (summary["n_units"], summary["t"], summary["psi"]) == (3, 3, 2)
    assert summary["statistics"]["outside_mass_mean"]["mean"] == pytest.approx((1.3 / 3 + 1) / 3)
    partitions = pd.read_parquet(output / "support_partitions.parquet")
    assert partitions.partition.tolist() == [0, 1, 2] * 3
    assert partitions.unit_id.tolist() == ["unit_zeta"] * 3 + ["unit_alpha"] * 3 + ["unit_beta"] * 3
    np.testing.assert_allclose(partitions.outside_mass, [0, 0.3, 1, 1, 1, 1, 0, 0, 0])
    assert (output / "support_group_summary.parquet").exists() == grouped
    if grouped:
        groups = pd.read_parquet(output / "support_group_summary.parquet")
        assert len(groups) == 2 * len(SUPPORT_METRICS)
        assert groups.group_value.unique().tolist() == ["A", "B"]
    config = json.loads((output / "config.json").read_text())
    assert config["command"] == "support"
    assert config["support"]["group_by"] == ("group" if grouped else None)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["unit_roles"]["analysis"] == table.unit_id.tolist()
    assert CliRunner().invoke(main, args).exit_code == 2  # Existing outputs are protected.


def test_unknown_group_leaves_no_partial_output(synthetic_support_artifact, tmp_path):
    output = tmp_path / "invalid"
    result = CliRunner().invoke(
        main,
        [
            "support",
            str(synthetic_support_artifact),
            "--output",
            str(output),
            "--group-by",
            "missing",
        ],
    )
    assert result.exit_code == 2
    assert "Unknown group-by column: missing" in result.output
    assert not output.exists()
