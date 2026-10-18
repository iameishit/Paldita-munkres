"""Hostile input and static guarantees: the package never executes, shells out or connects."""

import ast
import pathlib
import pickle
import pickletools
import subprocess
import sys

import pytest

import munkres
from munkres import DISALLOWED, Munkres, solve

SRC = pathlib.Path(munkres.__file__).parent
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "system",
    "popen",
    "Popen",
    "run",
    "check_output",
    "call",
}
FORBIDDEN_IMPORTS = {
    "socket",
    "subprocess",
    "ctypes",
    "urllib",
    "http",
    "requests",
    "marshal",
    "shelve",
    "ssl",
    "ftplib",
    "telnetlib",
    "smtplib",
}


def trees():
    for path in SRC.glob("*.py"):
        yield path.name, ast.parse(path.read_text(encoding="utf-8"))


def test_no_dangerous_calls_in_the_package():
    for name, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                assert called not in FORBIDDEN_CALLS, f"{name}:{node.lineno} calls {called}()"


def test_no_network_or_process_imports_in_the_package():
    for name, tree in trees():
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module.split(".")[0] not in FORBIDDEN_IMPORTS, f"{name} imports {module}"


def test_pickled_sentinel_only_references_a_global():
    opcodes = {op.name for op, _, _ in pickletools.genops(pickle.dumps(DISALLOWED, protocol=4))}
    assert opcodes <= {
        "PROTO",
        "FRAME",
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "MEMOIZE",
        "STACK_GLOBAL",
        "GLOBAL",
        "STOP",
    }


@pytest.mark.parametrize(
    "payload",
    [
        [[10**5000, 1], [1, 10**5000]],
        [[-(10**5000), 1], [1, 10**5000]],
        [[2**64] * 3] * 3,
    ],
)
def test_enormous_integers_are_just_numbers(payload):
    assert Munkres().compute(payload)


def test_self_referential_and_junk_structures_are_rejected_not_followed():
    loop = [1, 2]
    loop.append(loop)
    for bad in (
        [loop, [1, 2, 3]],
        [[object(), 1]],
        [[b"bytes", 1]],
        [[[1], [2]]],
        {1: [2]},
        {"a": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            Munkres().compute(bad)


def test_non_finite_input_cannot_hang(fixture_file):
    for bad in ([[float("nan")] * 3] * 3, [[-float("inf")] * 3] * 3):
        with pytest.raises(ValueError, match=r"."):
            solve(bad)
    assert solve([[float("inf"), 1], [1, float("inf")]]).total == 2


def test_cli_handles_hostile_files(tmp_path):
    cases = {
        "binary.csv": bytes(range(256)),
        "empty.csv": b"",
        "huge-cell.csv": b"1," + b"9" * 4000 + b"\n2,3\n",
        "junk.csv": b"1,2\n3,<script>alert(1)</script>\n",
    }
    for name, content in cases.items():
        path = tmp_path / name
        path.write_bytes(content)
        done = subprocess.run(
            [sys.executable, "-m", "munkres", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert done.returncode in (0, 2), (name, done.returncode, done.stderr)
        assert "Traceback" not in done.stderr
    assert (
        subprocess.run(
            [sys.executable, "-m", "munkres", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).returncode
        == 2
    )
