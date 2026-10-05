"""Behavioural tests for everything the bug-regression suite does not touch:
public helpers, error messages, input validation, mocks and smoke tests."""

import copy
import os
import pickle
import runpy
import subprocess
import sys
import textwrap
from decimal import Decimal
from fractions import Fraction
from unittest import mock

import pytest

import munkres
from munkres import DISALLOWED, Munkres, UnsolvableMatrix, make_cost_matrix, print_matrix

from .helpers import assert_valid_assignment, total

D = DISALLOWED
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


# -- DISALLOWED --------------------------------------------------------------


def test_disallowed_is_a_singleton_with_nice_repr():
    assert munkres.DISALLOWED_OBJ() is DISALLOWED
    assert repr(DISALLOWED) == "DISALLOWED"
    assert copy.copy(DISALLOWED) is DISALLOWED
    assert copy.deepcopy([DISALLOWED])[0] is DISALLOWED
    assert pickle.loads(pickle.dumps({"x": DISALLOWED}))["x"] is DISALLOWED


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_disallowed_identity_survives_every_pickle_protocol(protocol):
    clone = pickle.loads(pickle.dumps([DISALLOWED, [DISALLOWED]], protocol))
    assert clone[0] is DISALLOWED and clone[1][0] is DISALLOWED


# -- UnsolvableMatrix --------------------------------------------------------


def test_unsolvable_defaults_and_pickling():
    exc = UnsolvableMatrix()
    assert str(exc) == "Matrix cannot be solved!"
    assert exc.rows == () and exc.cols == ()
    with pytest.raises(UnsolvableMatrix):
        Munkres().compute([[D, D, 0], [D, D, 0], [0, 0, 0]])
    try:
        Munkres().compute([[D, D, 0], [D, D, 0], [0, 0, 0]])
    except UnsolvableMatrix as bad:
        clone = pickle.loads(pickle.dumps(bad))  # crosses process borders
        assert (clone.rows, clone.cols) == (bad.rows, bad.cols) == ((0, 1), (2,))
        assert str(clone) == str(bad)


@pytest.mark.parametrize(
    "matrix,message",
    [
        ([[D, D], [1, 2]], "Row 0 is entirely DISALLOWED."),
        ([[D, 1], [D, 2], [D, 3]], "Column 0 is entirely DISALLOWED."),
        (
            [[1, D, D], [1, D, D], [1, 2, 3]],
            "rows [0, 1] can only be matched to columns [0] (2 rows compete for 1 column)",
        ),
        (
            [[D, D], [D, D], [4, 9]],
            "columns [0, 1] can only be matched to rows [2] (2 columns compete for 1 row)",
        ),
        (
            [[1, 2, D, D], [3, 4, D, D], [5, 6, D, D], [1, 1, 1, 1]],
            "rows [0, 1, 2] can only be matched to columns [0, 1] (3 rows compete for 2 columns)",
        ),
    ],
)
def test_unsolvable_messages_name_the_conflict(matrix, message):
    with pytest.raises(UnsolvableMatrix) as info:
        Munkres().compute(matrix)
    assert message in str(info.value)


# -- input validation --------------------------------------------------------


@pytest.mark.parametrize(
    "bad,exc",
    [
        ("abc", TypeError),
        (5, TypeError),
        ([1, 2, 3], TypeError),  # 1-D
        ({0: [1, 2]}, TypeError),
        ([b"ab", b"cd"], TypeError),
        ([[1, "x"], [2, 3]], TypeError),
        ([[1, None], [2, 3]], TypeError),
        ([[1j, 1], [1, 2]], TypeError),
        ([[1, 2], [3]], ValueError),
        ([[float("nan"), 1], [2, 3]], ValueError),
        ([[Decimal("NaN"), 1], [2, 3]], ValueError),
        ([[Decimal("sNaN"), 1], [2, 3]], ValueError),  # used to leak InvalidOperation
        ([[-float("inf"), 1], [2, 3]], ValueError),
        ([[Decimal("-Infinity"), 1], [2, 3]], ValueError),
    ],
)
def test_invalid_input_is_rejected_cleanly(bad, exc):
    with pytest.raises(exc):
        Munkres().compute(bad)


def test_error_messages_say_where_the_problem_is():
    with pytest.raises(ValueError, match=r"row 1 has 1 entries but row 0 has 2"):
        Munkres().compute([[1, 2], [3]])
    with pytest.raises(TypeError, match=r"cell \[1\]\[1\] is not a number: None"):
        Munkres().compute([[1, 2], [3, None]])
    with pytest.raises(ValueError, match=r"cell \[0\]\[1\] is NaN"):
        Munkres().compute([[1, float("nan")], [3, 4]])


def test_generators_and_exotic_numeric_types():
    assert Munkres().compute(iter([[1, 2], [3, 4]])) == [(0, 0), (1, 1)]
    assert Munkres().compute([[True, False], [False, True]]) == [(0, 1), (1, 0)]
    frac = [[Fraction(1, 3), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 3)]]
    assert total(frac, Munkres().compute(frac)) == Fraction(2, 3)
    dec = [[Decimal("0.1"), D], [Decimal("0.2"), Decimal("0.1")]]
    assert total(dec, Munkres().compute(dec)) == Decimal("0.2")
    assert Munkres().compute([[Decimal("Infinity"), 1], [1, 2]]) == [(0, 1), (1, 0)]
    assert Munkres().compute([[0, 2**2000], [0, 2**2000]]) in ([(0, 0), (1, 1)], [(0, 1), (1, 0)])


def test_extreme_float_magnitudes_never_hang_or_misreport():
    for matrix in (
        [[1e308, -1e308], [-1e308, 1e308]],
        [[1e300, 1], [1, 1e300]],
        [[5e-324, 1e-300], [1e-300, 5e-324]],
    ):
        assert_valid_assignment(matrix, Munkres().compute(matrix))


# -- pad_matrix --------------------------------------------------------------


def test_pad_matrix_shapes_and_purity():
    m = Munkres()
    assert m.pad_matrix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    assert m.pad_matrix([[1, 2, 3], [4, 5, 6]], 9) == [[1, 2, 3], [4, 5, 6], [9, 9, 9]]
    assert m.pad_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 2, 0], [3, 4, 0], [5, 6, 0]]
    assert m.pad_matrix(((1,), (2, 3))) == [[1, 0], [2, 3]]  # ragged is padded
    assert m.pad_matrix([]) == []
    src = [[1, 2], [3, 4], [5, 6]]
    m.pad_matrix(src, 7)
    assert src == [[1, 2], [3, 4], [5, 6]]


# -- make_cost_matrix --------------------------------------------------------


def test_make_cost_matrix_variants():
    assert make_cost_matrix([[1, 2], [3, 4]]) == [[3, 2], [1, 0]]
    assert make_cost_matrix([[1, 2], [3, 4]], lambda x: 10 - x) == [[9, 8], [7, 6]]
    assert make_cost_matrix(((1, 2), (3, 4))) == [[3, 2], [1, 0]]
    assert make_cost_matrix([]) == []
    assert make_cost_matrix([[D, D]]) == [[D, D]]  # nothing to invert
    calls = []
    out = make_cost_matrix([[5, D], [float("inf"), 2]], lambda x: calls.append(x) or -x)
    assert out == [[-5, D], [float("inf"), -2]] and sorted(calls) == [2, 5]
    # the +inf cell must not poison the default maximum
    assert make_cost_matrix([[5, float("inf")], [2, 3]]) == [[0, float("inf")], [3, 2]]


def test_make_cost_matrix_round_trips_through_solver():
    profit = [[5, 9, 1], [10, 3, 2], [8, 7, 4]]
    result = Munkres().compute(make_cost_matrix(profit))
    assert total(profit, result) == 23  # brute-force maximum


# -- print_matrix (stdout is captured) ---------------------------------------


def test_print_matrix_output(capsys):
    print_matrix([[1, D], [2.5, 3]], msg="hello")
    assert capsys.readouterr().out == "hello\n[  1,   D]\n[2.5,   3]\n"
    print_matrix(((1, 2),))
    assert capsys.readouterr().out == "[1, 2]\n"
    print_matrix([])
    assert capsys.readouterr().out == ""


# -- the demo / __main__ -----------------------------------------------------


def test_module_demo_runs_and_checks_itself(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["munkres"])  # don't leak pytest's own flags
    monkeypatch.delitem(sys.modules, "munkres.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("munkres", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert out.count("lowest cost=") == 10 and "cost matrix" in out


# -- mocks: optional dependencies are really optional ------------------------


def test_works_without_numpy_or_pandas_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert Munkres().compute([[4, 1], [2, 0]]) == [(0, 1), (1, 0)]
    assert make_cost_matrix([[1, 2]]) == [[1, 0]]


def test_package_imports_with_no_third_party_modules():
    code = textwrap.dedent("""
        import sys
        sys.modules['numpy'] = None; sys.modules['pandas'] = None
        sys.modules['scipy'] = None
        import munkres
        assert munkres.Munkres().compute([[1, 2], [3, 4]]) == [(0, 0), (1, 1)]
        third_party = {m.split('.')[0] for m in sys.modules} & {'numpy', 'pandas', 'scipy'}
        assert all(sys.modules[m] is None for m in third_party)
        print('ok')
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    assert out.stdout.strip() == "ok"


class _FakeFrame:
    """Duck-typed DataFrame: only to_numpy() is used."""

    def __init__(self, data):
        self._data = data

    def to_numpy(self):
        return self._data


def test_dataframe_like_objects_are_accepted_and_untouched():
    np_ = pytest.importorskip("numpy")
    data = np_.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    keep = data.copy()
    assert Munkres().compute(_FakeFrame(data)) == [(0, 1), (1, 0), (2, 2)]
    assert (data == keep).all()


def test_real_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    snapshot = frame.copy()
    assert Munkres().compute(frame) == [(0, 1), (1, 0), (2, 2)]
    assert frame.equals(snapshot)


def test_hang_guard_falls_back_to_a_thread_without_sigalrm():
    """The helper must still catch hangs where SIGALRM does not exist (Windows)."""
    import time

    from . import helpers

    with mock.patch.object(helpers, "signal", mock.Mock(spec=[])):
        assert helpers.call_with_timeout(lambda: 42, 1) == 42
        with pytest.raises(ValueError):
            helpers.call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("x")), 1)
        with pytest.raises(helpers.Hang):
            helpers.call_with_timeout(time.sleep, 0.2, 5)


# -- smoke ------------------------------------------------------------------


def test_readme_style_quickstart():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = Munkres().compute(cost)
    assert pairs == [(0, 1), (1, 0), (2, 2)]
    assert sum(cost[r][c] for r, c in pairs) == 5


# ---------------------------------------------------------------------------
# `python -m munkres` (the installation self-check)
# ---------------------------------------------------------------------------


def test_module_main_runs_its_self_check(capsys, monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["munkres"])  # don't leak pytest's own flags
    monkeypatch.delitem(sys.modules, "munkres.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("munkres", run_name="__main__")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "lowest cost=850" in out and "lowest cost=20.028" in out


def test_module_main_fails_loudly_if_an_answer_is_wrong(monkeypatch):
    import munkres.__main__ as entry

    monkeypatch.setattr(entry, "EXAMPLES", [([[1, 2], [3, 4]], 999)])
    with pytest.raises(SystemExit, match="self-check failed: expected 999, got 5"):
        entry.main()
