"""Phase 4 API: solve / Assignment / gating / maximize / trace / diagnose /
linear_sum_assignment / build_cost_matrix. Checked against independent oracles."""

import dataclasses
import itertools
import random
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import munkres
from munkres import (
    DISALLOWED,
    Assignment,
    Munkres,
    UnsolvableMatrix,
    build_cost_matrix,
    diagnose,
    linear_sum_assignment,
    solve,
)

from .helpers import brute_force_min_cost, call_with_timeout, scipy_min_cost

D = DISALLOWED
INF = float("inf")
COMMON = {
    "deadline": None,
    "database": None,
    "derandomize": True,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}
ints = st.integers(-30, 30)


def matrices(max_rows=5, max_cols=5, *, forbid=False):
    cell = st.one_of(ints, st.just(D)) if forbid else ints

    @st.composite
    def build(draw):
        r, c = draw(st.integers(1, max_rows)), draw(st.integers(1, max_cols))
        return [[draw(cell) for _ in range(c)] for _ in range(r)]

    return build()


def brute_force_gain(matrix, threshold):
    """Best total of (threshold - cost) over every partial matching that only
    uses allowed pairs costing <= threshold. Exhaustive: tiny matrices only."""
    rows, cols = len(matrix), len(matrix[0])

    def best(i, used):
        if i == rows:
            return 0
        top = best(i + 1, used)  # leave row i unmatched
        for j in range(cols):
            v = matrix[i][j]
            if j not in used and v is not D and v <= threshold:
                top = max(top, (threshold - v) + best(i + 1, used | {j}))
        return top

    return best(0, frozenset())


# --------------------------------------------------------------------------
# solve(): basics and the Assignment object
# --------------------------------------------------------------------------


@settings(max_examples=150, **COMMON)
@given(matrices(forbid=True))
def test_solve_agrees_with_compute_and_oracle(matrix):
    expected = brute_force_min_cost(matrix)
    if expected is None:
        with pytest.raises(UnsolvableMatrix):
            solve(matrix)
        return
    result = solve(matrix)
    assert result.total == expected
    assert list(result.pairs) == Munkres().compute(matrix)


def test_assignment_object_api():
    result = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert isinstance(result, Assignment)
    assert result.pairs == ((0, 1), (1, 0), (2, 2))
    assert list(result) == list(result.pairs) and len(result) == 3
    assert result.rows == (0, 1, 2) and result.cols == (1, 0, 2)
    assert result.as_dict() == {0: 1, 1: 0, 2: 2}
    assert result.total == 5 and result.shape == (3, 3) and result.maximize is False
    assert result.unmatched_rows == () and result.unmatched_cols == ()
    assert result.labelled() == [(0, 1), (1, 0), (2, 2)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.total = 0  # type: ignore[misc]


def test_unmatched_reporting_wide_and_tall():
    wide = solve([[4, 1, 3], [2, 0, 5]])
    assert wide.unmatched_rows == () and wide.unmatched_cols == (2,)
    tall = solve([[1, 2], [3, 4], [0, 9]])
    assert tall.unmatched_rows == (1,) and tall.unmatched_cols == ()
    assert tall.total == 2


def test_empty_assignment_is_falsy_and_valid():
    for empty in ([], [[]], [[], []]):
        result = solve(empty)
        assert not result and result.pairs == () and result.total == 0


# --------------------------------------------------------------------------
# maximize
# --------------------------------------------------------------------------


@settings(max_examples=200, **COMMON)
@given(matrices(forbid=True))
def test_maximize_matches_oracle(profit):
    negated = [[v if v is D else -v for v in row] for row in profit]
    expected = brute_force_min_cost(negated)
    if expected is None:
        with pytest.raises(UnsolvableMatrix):
            solve(profit, maximize=True)
    else:
        result = solve(profit, maximize=True)
        assert result.total == -expected and result.maximize is True


def test_maximize_infinities():
    assert solve([[-INF, 5], [7, 1]], maximize=True).pairs == ((0, 1), (1, 0))  # -inf forbids
    with pytest.raises(ValueError, match=r"\+infinity"):
        solve([[INF, 1], [1, 1]], maximize=True)
    with pytest.raises(ValueError, match="NaN"):
        solve([[float("nan"), 1], [1, 1]], maximize=True)


def test_maximize_is_exact_for_decimal_and_fraction():
    dec = [[Decimal("0.1"), Decimal("0.2")], [Decimal("0.2"), Decimal("0.1")]]
    assert solve(dec, maximize=True).total == Decimal("0.4")
    fr = [[Fraction(1, 3), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 3)]]
    assert solve(fr, maximize=True).total == Fraction(1)


def test_maximize_rejects_signalling_nan_and_non_numbers():
    with pytest.raises(ValueError, match="NaN"):
        solve([[Decimal("sNaN"), 1], [1, 1]], maximize=True)
    with pytest.raises(TypeError, match="not a number"):
        solve([["x", 1], [1, 1]], maximize=True)


def test_gate_flags_must_match_the_mode():
    with pytest.raises(ValueError, match="min_profit"):
        solve([[1]], maximize=True, max_cost=1)
    with pytest.raises(ValueError, match="maximize=True"):
        solve([[1]], min_profit=1)


# --------------------------------------------------------------------------
# gating (max_cost / min_profit)
# --------------------------------------------------------------------------


def check_gated(matrix, threshold, result):
    """Invariants every gated answer must satisfy, then optimality vs brute force."""
    rows, cols = len(matrix), len(matrix[0])
    assert len({r for r, _ in result.pairs}) == len(result.pairs)
    assert len({c for _, c in result.pairs}) == len(result.pairs)
    for r, c in result.pairs:
        assert matrix[r][c] is not D and matrix[r][c] <= threshold
    assert set(result.unmatched_rows) == set(range(rows)) - {r for r, _ in result.pairs}
    assert set(result.unmatched_cols) == set(range(cols)) - {c for _, c in result.pairs}
    gain = sum(threshold - matrix[r][c] for r, c in result.pairs)
    assert gain == brute_force_gain(matrix, threshold)


@settings(max_examples=300, **COMMON)
@given(matrices(forbid=True), st.integers(-35, 35))
def test_max_cost_gating_is_optimal_and_never_exceeds_threshold(matrix, threshold):
    result = call_with_timeout(lambda: solve(matrix, max_cost=threshold), 5)
    check_gated(matrix, threshold, result)
    assert result.total == sum(matrix[r][c] for r, c in result.pairs)


@settings(max_examples=200, **COMMON)
@given(matrices(forbid=True), st.integers(-35, 35))
def test_min_profit_gating_mirrors_max_cost(profit, floor):
    result = solve(profit, maximize=True, min_profit=floor)
    negated = [[v if v is D else -v for v in row] for row in profit]
    check_gated(negated, -floor, solve(negated, max_cost=-floor))
    for r, c in result.pairs:
        assert profit[r][c] >= floor
    assert result.total == sum(profit[r][c] for r, c in result.pairs)


def test_gating_can_never_be_unsolvable():
    hall_violation = [[1, D, D], [1, D, D], [1, 2, 3]]
    with pytest.raises(UnsolvableMatrix):
        solve(hall_violation)
    result = solve(hall_violation, max_cost=10)
    assert len(result) == 2 and result.total == 3  # (0,0) or (1,0) cost 1, plus (2,1) cost 2


def test_gating_examples():
    m = [[1, 9, 9], [9, 2, 9], [9, 9, 8]]
    result = solve(m, max_cost=3)
    assert result.pairs == ((0, 0), (1, 1)) and result.unmatched_rows == (2,)
    assert solve(m, max_cost=100).pairs == ((0, 0), (1, 1), (2, 2))  # loose gate = no gate
    assert solve(m, max_cost=0).pairs == ()  # nothing is cheap enough
    tall = solve([[1, 9], [9, 1], [2, 2]], max_cost=1)
    assert tall.pairs == ((0, 0), (1, 1)) and tall.unmatched_rows == (2,)


def test_gate_extremes_and_types():
    m = [[1, 2], [2, 1]]
    assert solve(m, max_cost=INF).pairs == ((0, 0), (1, 1))  # +inf = no gate
    assert solve(m, max_cost=-INF).pairs == ()  # -inf = nothing allowed
    assert solve(m, maximize=True, min_profit=INF).pairs == ()
    assert solve(m, maximize=True, min_profit=-INF).total == 4
    assert solve([[Decimal("0.5")]], max_cost=Decimal("1")).pairs == ((0, 0),)
    with pytest.raises(TypeError, match="max_cost"):
        solve(m, max_cost="3")
    with pytest.raises(ValueError, match="NaN"):
        solve(m, max_cost=float("nan"))


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------


def test_diagnose():
    assert diagnose([[1, 2], [3, 4]]) is None
    found = diagnose([[1, D, D], [1, D, D], [1, 2, 3]])
    assert found.rows == (0, 1) and found.cols == (0,) and "compete" in found.message
    tall = diagnose([[D, D], [D, D], [4, 9]])
    assert tall.cols == (0, 1) and tall.rows == (2,)
    forbidden_col = diagnose([[-INF, 1], [-INF, 2]], maximize=True)  # -inf profit = forbidden
    assert forbidden_col.rows == (0, 1) and forbidden_col.cols == (1,)
    assert diagnose([[1, 2], [3, 4]], maximize=True) is None


# --------------------------------------------------------------------------
# linear_sum_assignment: a SciPy drop-in
# --------------------------------------------------------------------------


def test_matches_scipy_exactly_when_the_optimum_is_unique():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = random.Random(11)
    for _ in range(300):
        r, c = rng.randint(1, 9), rng.randint(1, 9)
        m = [[rng.random() * 100 for _ in range(c)] for _ in range(r)]
        for maximize in (False, True):
            want_r, want_c = scipy_opt.linear_sum_assignment(m, maximize=maximize)
            got_r, got_c = linear_sum_assignment(m, maximize=maximize)
            assert got_r == list(want_r) and got_c == list(want_c)


def test_scipy_parity_on_infinities_and_infeasibility():
    scipy_opt = pytest.importorskip("scipy.optimize")
    cases = [[[INF, 1], [1, INF]], [[1, INF], [2, INF]], [[1, 2], [3, float("nan")]], [[1, -INF]]]
    for m in cases:
        for maximize in (False, True):
            try:
                want = scipy_opt.linear_sum_assignment(m, maximize=maximize)
            except ValueError:
                with pytest.raises(ValueError, match=r"."):
                    linear_sum_assignment(m, maximize=maximize)
            else:
                got = linear_sum_assignment(m, maximize=maximize)
                assert got[0] == list(want[0]) and got[1] == list(want[1])


def test_return_type_follows_the_input():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rows, cols = linear_sum_assignment(np.array([[4, 1], [2, 0]]))
    assert isinstance(rows, np.ndarray) and rows.dtype == np.intp and cols.tolist() == [1, 0]
    assert isinstance(linear_sum_assignment(pd.DataFrame([[4, 1], [2, 0]]))[0], np.ndarray)
    assert linear_sum_assignment(((4, 1), (2, 0))) == ([0, 1], [1, 0])
    assert linear_sum_assignment([]) == ([], [])
    empty_rows, _ = linear_sum_assignment(np.zeros((0, 3)))
    assert len(empty_rows) == 0


def test_errors_are_valueerrors_like_scipy():
    with pytest.raises(ValueError, match="compete"):
        linear_sum_assignment([[1, D, D], [1, D, D], [1, 2, 3]])
    with pytest.raises(ValueError, match="not a number"):
        linear_sum_assignment([["a"]])


class _FakeArray:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return [list(r) for r in self._rows]


def test_without_numpy_installed_lists_come_back(monkeypatch):
    import munkres._api as api

    def no_numpy(name, *args, **kwargs):
        raise ImportError(name)

    monkeypatch.setattr(api.importlib, "import_module", no_numpy)
    assert linear_sum_assignment(_FakeArray([[4, 1], [2, 0]])) == ([0, 1], [1, 0])


# --------------------------------------------------------------------------
# build_cost_matrix
# --------------------------------------------------------------------------


def test_build_cost_matrix():
    assert build_cost_matrix([0, 5], [1, 2], lambda a, b: abs(a - b)) == [[1, 2], [4, 3]]
    gen = (x for x in (0, 5))
    built = build_cost_matrix(gen, (y for y in (1, 9)), lambda a, b: None if b == 9 else a + b)
    assert built == [[1, D], [6, D]]
    assert build_cost_matrix([], [1, 2], lambda a, b: 0) == []


def test_build_cost_matrix_feeds_the_solver():
    workers, jobs = [(0, 0), (5, 5)], [(1, 1), (6, 4)]
    cost = build_cost_matrix(workers, jobs, lambda w, j: abs(w[0] - j[0]) + abs(w[1] - j[1]))
    assert solve(cost).pairs == ((0, 0), (1, 1))


# --------------------------------------------------------------------------
# labels: real pandas, and a mock that works without pandas installed
# --------------------------------------------------------------------------


def test_pandas_labels_roundtrip():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[4, 1], [2, 0]], index=["w1", "w2"], columns=["jobA", "jobB"])
    result = solve(df)
    assert result.labelled() == [("w1", "jobB"), ("w2", "jobA")]
    assert result.row_labels == ("w1", "w2") and result.col_labels == ("jobA", "jobB")
    assert df.loc["w1", "jobA"] == 4  # input untouched


class _FakeIndex(list):
    def tolist(self):
        return list(self)


class _FakeFrame:
    def __init__(self, rows, index, columns):
        self._rows = rows
        self.index, self.columns = _FakeIndex(index), _FakeIndex(columns)

    def to_numpy(self):
        return _FakeArray(self._rows)


def test_dataframe_like_objects_work_without_pandas():
    frame = _FakeFrame([[4, 1], [2, 0]], ["a", "b"], ["x", "y"])
    assert solve(frame).labelled() == [("a", "y"), ("b", "x")]


def test_a_plain_list_is_not_mistaken_for_a_frame():
    # list has an `.index` *method*; it must not be treated as row labels
    assert solve([[4, 1], [2, 0]]).row_labels is None


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


def test_public_api_is_exported():
    for name in (
        "solve", "Assignment", "Diagnosis", "Trace", "diagnose",
        "linear_sum_assignment", "build_cost_matrix",
    ):  # fmt: skip
        assert name in munkres.__all__ and hasattr(munkres, name)


def test_solve_is_consistent_with_the_exhaustive_oracle_on_rectangles():
    rng = random.Random(5)
    for r, c in itertools.product(range(1, 6), repeat=2):
        for _ in range(10):
            m = [[rng.randint(-9, 9) for _ in range(c)] for _ in range(r)]
            assert solve(m).total == scipy_min_cost(m)


def test_an_infinite_gate_means_no_gate_even_for_exact_number_types():
    """+inf is compared, never used in arithmetic, so Decimal and Fraction keep working."""
    exact_decimal = [[Decimal("1.5"), Decimal("2")], [Decimal("2"), Decimal("1")]]
    assert solve(exact_decimal, max_cost=INF).total == Decimal("2.5")
    exact_fraction = [[Fraction(1, 3), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 3)]]
    assert solve(exact_fraction, max_cost=INF).total == Fraction(2, 3)
    assert solve(exact_decimal, maximize=True, min_profit=-INF).total == Decimal("4")
    # both rows want column 0, which forces a real augmenting search (not just the warm start)
    contested = [[Decimal(1), Decimal(5)], [Decimal(1), Decimal(6)]]
    assert solve(contested, max_cost=INF).total == Decimal(6)
