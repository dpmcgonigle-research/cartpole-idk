import sys

import numpy as np
import pytest

from cartpole_idk.cli.prepare import main
from cartpole_idk.preparation import prepare_dataset
from cartpole_idk.storage import Trajectory, TrajectoryStore


def make_dataset(path, lengths):
    store = TrajectoryStore(path)
    for i, length in enumerate(lengths):
        states = np.arange((length + 1) * 4).reshape(-1, 4)
        store.add(
            Trajectory(
                str(i),
                states,
                states + 1,
                np.arange(length) % 2,
                np.arange(length) % 2,
                np.ones(length),
                np.arange(length) == length - 1,
                np.zeros(length, dtype=bool),
                {"perturbation_onset": 20, "perturbation": {"onset": 20}},
            )
        )
    return store


def test_failure_tail_and_minimum(tmp_path):
    source = make_dataset(tmp_path / "source", [24, 25, 99, 100, 299, 300])
    prepared = prepare_dataset(source.root, tmp_path / "prepared")
    assert list(prepared.manifest().trajectory_id) == ["1", "2", "3", "4", "5"]
    for tid, start, stop in [("1", 0, 25), ("2", 0, 99), ("3", 0, 100), ("4", 199, 299)]:
        actual, original = prepared.get(tid), source.get(tid)
        assert actual.metadata["success"] is False
        assert actual.metadata["source_length"] == stop
        assert actual.metadata["segment_start"] == start
        for field in ["true_observations", "agent_observations"]:
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(original, field)[start : stop + 1]
            )
        for field in [
            "commanded_actions",
            "executed_actions",
            "rewards",
            "terminated",
            "truncated",
        ]:
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(original, field)[start:stop]
            )
        assert actual.metadata["perturbation_onset"] == 20 - start
        assert actual.metadata["perturbation"]["onset"] == 20 - start
    assert prepared.get("5").metadata["success"] is True


def test_uniform_success_starts_and_reproducibility(tmp_path):
    source = make_dataset(tmp_path / "source", [300, 301, 500] * 4)
    expected = np.random.default_rng(42).integers(0, 151, size=12)
    for name in ["first", "second"]:
        store = prepare_dataset(source.root, tmp_path / name, seed=42)
        np.testing.assert_array_equal(store.manifest().segment_start, expected)
        for tid in store.manifest().trajectory_id:
            traj = store.get(tid)
            assert traj.length == 100
            assert traj.metadata["segment_stop"] <= 250


@pytest.mark.parametrize("start", [0, 150])
def test_fixed_start(tmp_path, start):
    source = make_dataset(tmp_path / "source", [25, 160, 175, 299, 300, 500])
    store = prepare_dataset(source.root, tmp_path / "prepared", episode_start=start)
    assert set(store.manifest().segment_start) == {start}
    assert list(store.manifest().length) == (
        [25, 100, 100, 100, 100, 100] if start == 0 else [25, 100, 100, 100]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_length": 0},
        {"min_episode_length": 0},
        {"min_episode_length": 101},
        {"success_threshold": 0},
        {"success_buffer": -1},
        {"success_buffer": 300},
        {"episode_length": 251},
        {"episode_start": -1},
        {"episode_start": 151},
        {"seed": -1},
    ],
)
def test_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare_dataset(tmp_path / "source", tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_exact_fit_and_empty_output(tmp_path):
    source = make_dataset(tmp_path / "source", [24, 300])
    store = prepare_dataset(source.root, tmp_path / "exact", episode_length=250)
    assert store.get("1").metadata["segment_start"] == 0
    short = make_dataset(tmp_path / "short", [24])
    assert prepare_dataset(short.root, tmp_path / "empty").manifest().empty


def test_cli_and_protect_existing_data(tmp_path, monkeypatch):
    source = make_dataset(tmp_path / "source", [300])
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["cartpole-prepare", str(source.root), "--output", str(output)]
    )
    main(standalone_mode=False)
    assert TrajectoryStore(output).get("0").length == 100
    for destination in [source.root, output]:
        with pytest.raises(ValueError):
            prepare_dataset(source.root, destination)


def test_multiple_datasets_cli(tmp_path, monkeypatch):
    import json

    first = make_dataset(tmp_path / "first", [300, 24])
    second = make_dataset(tmp_path / "second", [299, 500])
    output = tmp_path / "combined"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cartpole-prepare",
            str(first.root),
            str(second.root),
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    main(standalone_mode=False)
    store = TrajectoryStore(output)
    rows = store.manifest()
    assert len(rows) == 3
    assert rows.trajectory_id.is_unique
    assert len(list(store.trajectory_dir.glob("*.npz"))) == 3
    expected_starts = iter(np.random.default_rng(42).integers(0, 151, size=2))
    for tid in rows.trajectory_id:
        segment = store.get(tid)
        meta = segment.metadata
        source = TrajectoryStore(meta["source_dataset"]).get(meta["source_trajectory_id"])
        start = next(expected_starts) if source.length >= 300 else 199
        assert meta["segment_start"] == start
        np.testing.assert_array_equal(
            segment.true_observations, source.true_observations[start : start + 101]
        )
    report = json.loads((output / "preparation_report.json").read_text())
    assert report["source_datasets"] == [str(first.root), str(second.root)]
    assert (report["saved"], report["skipped"]) == (3, 1)
    repeated = prepare_dataset([first.root, second.root], tmp_path / "repeat", seed=42)
    assert repeated.manifest().segment_start.tolist() == rows.segment_start.tolist()


def test_multiple_sources_validated_before_writing(tmp_path):
    source = make_dataset(tmp_path / "source", [300])
    output = tmp_path / "out"
    for inputs in [[], [source.root, source.root / "."], [source.root, tmp_path / "missing"]]:
        with pytest.raises(ValueError):
            prepare_dataset(inputs, output)
        assert not output.exists()
    other = make_dataset(tmp_path / "other", [300])
    with pytest.raises(ValueError, match="every source"):
        prepare_dataset([source.root, other.root], other.root)


def test_multiple_sources_all_skipped(tmp_path):
    sources = [make_dataset(tmp_path / name, [24]).root for name in ["first", "second"]]
    assert prepare_dataset(sources, tmp_path / "out").manifest().empty
