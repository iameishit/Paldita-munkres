"""Every example and benchmark script runs and prints something sensible."""

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
        cwd=ROOT,
    )


def test_all_expected_scripts_exist():
    assert {p.name for p in EXAMPLES} == {
        "basic.py", "rectangular.py", "maximize.py", "disallowed.py",
        "numpy.py", "analysis.py", "trace.py", "large_matrix.py",
    }  # fmt: skip
    assert len(BENCHMARKS) == 6


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    if script.name == "numpy.py":
        pytest.importorskip("numpy")
    done = run(script)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() and "Traceback" not in done.stderr


def test_basic_example_prints_the_known_answer():
    out = run(ROOT / "examples" / "basic.py").stdout
    assert "[(0, 1), (1, 0), (2, 2)]" in out and "total cost: 5" in out


@pytest.mark.parametrize("script", BENCHMARKS, ids=lambda p: p.name)
def test_benchmark_runs_in_quick_mode(script):
    done = run(script, "--quick")
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.strip().splitlines()) >= 2


def test_release_verifier_reports_no_problems():
    done = run(ROOT / "tools" / "verify_release.py")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "RELEASE CHECKS PASSED" in done.stdout
