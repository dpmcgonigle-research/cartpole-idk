import json

import pytest

from cartpole_idk.model import GeneratedTrajectory, GenerationReport, ReturnStatistics


def test_return_statistics():
    stats = ReturnStatistics.from_returns([1, 2, 3, 4])
    assert stats is not None
    assert stats.model_dump() == {
        "mean": 2.5,
        "std_dev": pytest.approx(1.11803398875),
        "min": 1,
        "max": 4,
        "q1": 1.75,
        "median": 2.5,
        "q3": 3.25,
    }
    assert ReturnStatistics.from_returns([]) is None
    single = ReturnStatistics.from_returns([5])
    assert single is not None
    assert single.std_dev == 0
    assert single.q1 == single.median == single.q3 == 5


def test_generation_report_roundtrip_and_legacy(tmp_path):
    report = GenerationReport(
        checkpoint="checkpoint.pt",
        episodes=1,
        seed=42,
        perturbation_type="none",
        generated=[GeneratedTrajectory(trajectory_id="one", episode_return=10, length=10)],
    )
    path = tmp_path / "report.json"
    report.to_file(path)
    raw = json.loads(path.read_text())
    assert raw["generated"][0]["return"] == 10
    assert raw["return_statistics"]["mean"] == 10
    assert GenerationReport.from_file(path) == report
    del raw["return_statistics"]
    path.write_text(json.dumps(raw))
    assert GenerationReport.from_file(path) == report
    report.generated.clear()
    report.to_file(path)
    assert json.loads(path.read_text())["return_statistics"] is None
