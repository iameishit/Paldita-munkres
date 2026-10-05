"""Every example and benchmark script must run, so the documentation never goes stale."""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
BENCHMARKS = sorted((ROOT / "benchmarks").glob("benchmark_*.py"))


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_and_prints(script):
    if script.name == "numpy.py":
        pytest.importorskip("numpy")
    done = run(script)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()
    assert "Traceback" not in done.stderr


@pytest.mark.parametrize("script", BENCHMARKS, ids=lambda p: p.name)
def test_benchmark_runs_in_quick_mode(script):
    done = run(script, "--quick")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()


def test_all_expected_scripts_are_present():
    assert len(EXAMPLES) == 8 and len(BENCHMARKS) == 6
