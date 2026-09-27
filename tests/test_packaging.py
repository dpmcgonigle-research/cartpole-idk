"""Source packages must not be excluded by Hatchling's Git-ignore filtering."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_modules_are_not_gitignored():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("Source-checkout ignore check requires Git")
    modules = sorted((ROOT / "src/cartpole_idk").rglob("*.py"))
    assert modules
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(str(path.relative_to(ROOT)) for path in modules) + "\n",
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert not result.stdout, f"Source modules excluded from distributions:\n{result.stdout}"
