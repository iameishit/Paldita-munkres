"""The command line interface, in-process (mocked stdin, tmp files) and as a real process."""

import io
import json
import shutil
import subprocess
import sys

import pytest

import munkres
from munkres import DISALLOWED as D
from munkres import cli

MATRIX = "4,1,3\n2,0,5\n3,2,2\n"


@pytest.fixture
def matrix_file(tmp_path):
    path = tmp_path / "costs.csv"
    path.write_text(MATRIX)
    return str(path)


# -- parsing -------------------------------------------------------------------


def test_parse_matrix_formats():
    assert cli.parse_matrix(MATRIX) == [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    assert cli.parse_matrix("4 1 3\n 2   0 5\n") == [[4, 1, 3], [2, 0, 5]]  # whitespace separated
    assert cli.parse_matrix("# costs\n\n1.5, 2\n") == [[1.5, 2]]  # comments, blanks, floats
    assert cli.parse_matrix("1,,3\nD,x,-\ninf, None ,+inf\n") == [
        [1, D, 3],
        [D, D, D],
        [D, D, D],
    ]
    assert cli.parse_matrix("") == []


def test_parse_matrix_reports_the_line_of_a_bad_cell():
    with pytest.raises(ValueError, match=r"line 2: cannot read 'abc'"):
        cli.parse_matrix("1,2\n3,abc\n")


# -- in-process runs -----------------------------------------------------------


def test_plain_output(matrix_file, capsys):
    assert cli.main([matrix_file]) == 0
    assert capsys.readouterr().out.splitlines() == ["0 1 1", "1 0 2", "2 2 2", "total: 5"]


def test_json_output_and_unmatched(tmp_path, capsys):
    path = tmp_path / "wide.csv"
    path.write_text("4,1,3\n2,0,5\n")
    assert cli.main([str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "pairs": [[0, 1], [1, 0]],
        "total": 3,
        "unmatched_rows": [],
        "unmatched_cols": [2],
        "shape": [2, 3],
        "maximize": False,
    }
    assert cli.main([str(path)]) == 0
    assert "unmatched columns: [2]" in capsys.readouterr().out


def test_unmatched_rows_are_printed(tmp_path, capsys):
    path = tmp_path / "tall.csv"
    path.write_text("1,2\n3,4\n0,9\n")
    assert cli.main([str(path)]) == 0
    assert "unmatched rows: [1]" in capsys.readouterr().out


def test_maximize_and_gates(matrix_file, tmp_path, capsys):
    assert cli.main([matrix_file, "--maximize"]) == 0
    assert "total: 11" in capsys.readouterr().out  # best profit (verified with scipy)
    path = tmp_path / "g.csv"
    path.write_text("1,9,9\n9,2,9\n9,9,8\n")
    assert cli.main([str(path), "--max-cost", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pairs"] == [[0, 0], [1, 1]]
    assert cli.main([str(path), "--maximize", "--min-profit", "5", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 27 and len(data["pairs"]) == 3  # two tied optima exist
    assert cli.main([str(path), "--maximize", "--min-profit", "10", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pairs"] == []  # nothing earns 10


def test_wrong_gate_flag_for_the_mode_is_an_error(matrix_file, capsys):
    assert cli.main([matrix_file, "--maximize", "--max-cost", "3"]) == 2
    assert "--max-cost does not apply" in capsys.readouterr().err
    assert cli.main([matrix_file, "--min-profit", "3"]) == 2
    assert "--min-profit does not apply" in capsys.readouterr().err


def test_reads_standard_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(MATRIX))
    assert cli.main(["-"]) == 0
    assert "total: 5" in capsys.readouterr().out


def test_trace_flags(matrix_file, tmp_path, capsys):
    assert cli.main([matrix_file, "--trace"]) == 0
    assert "Hungarian algorithm on a 3x3 matrix" in capsys.readouterr().out
    assert cli.main([matrix_file, "--trace", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["trace"][-1].startswith("Done.")
    page = tmp_path / "trace.html"
    assert cli.main([matrix_file, "--trace-html", str(page)]) == 0
    assert page.read_text().startswith("<!doctype html>")


def test_errors_have_distinct_exit_codes(tmp_path, capsys):
    bad = tmp_path / "hall.csv"
    bad.write_text("1,D,D\n1,D,D\n1,2,3\n")
    assert cli.main([str(bad)]) == 1  # unsolvable
    assert "compete" in capsys.readouterr().err
    junk = tmp_path / "junk.csv"
    junk.write_text("1,2\n3,oops\n")
    assert cli.main([str(junk)]) == 2  # unreadable input
    assert "line 2" in capsys.readouterr().err
    assert cli.main([str(tmp_path / "missing.csv")]) == 2  # no such file
    assert "error:" in capsys.readouterr().err
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1,2\n3\n")
    assert cli.main([str(ragged)]) == 2
    assert "rectangular" in capsys.readouterr().err
    assert cli.main(["--maximize"]) == 2  # flags but no file
    assert "give a matrix file" in capsys.readouterr().err


def test_no_arguments_runs_the_self_check(capsys):
    assert cli.main([]) == 0
    assert "lowest cost=850" in capsys.readouterr().out
    assert cli.main(["--self-check"]) == 0


def test_main_reads_sys_argv_by_default(monkeypatch, matrix_file, capsys):
    monkeypatch.setattr(sys, "argv", ["munkres", matrix_file])
    assert cli.main() == 0
    assert "total: 5" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert munkres.__version__ in capsys.readouterr().out


# -- as a real process ---------------------------------------------------------


def run(*args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "munkres", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_python_dash_m_munkres_end_to_end(matrix_file):
    done = run(matrix_file, "--json")
    assert done.returncode == 0 and json.loads(done.stdout)["total"] == 5
    assert run("-", stdin="1,D\nD,1\n").stdout.endswith("total: 2\n")
    failed = run("-", stdin="1,D\n1,D\n")
    assert failed.returncode == 1 and "error:" in failed.stderr and failed.stdout == ""
    assert run().returncode == 0  # self-check


@pytest.mark.skipif(shutil.which("munkres") is None, reason="console script not installed")
def test_console_script_is_installed(matrix_file):
    done = subprocess.run(
        ["munkres", matrix_file], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0 and "total: 5" in done.stdout
