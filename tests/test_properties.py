"""
Property-based tests (Hypothesis). The solver's answers are checked against
INDEPENDENT oracles (exhaustive search and scipy), never against itself.
"""

import copy

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from munkres import DISALLOWED, Munkres, UnsolvableMatrix, make_cost_matrix

from .helpers import (
    assert_valid_assignment,
    brute_force_min_cost,
    call_with_timeout,
    scipy_min_cost,
    total,
)

D = DISALLOWED

COMMON = dict(
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def solve(matrix, seconds=2):
    return call_with_timeout(Munkres().compute, seconds, matrix)


# -- strategies --------------------------------------------------------------

ints = st.integers(min_value=-50, max_value=50)
floats = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


def matrices(elements, max_rows=6, max_cols=6, forbid=False):
    cell = st.one_of(elements, st.just(D)) if forbid else elements

    @st.composite
    def build(draw):
        r = draw(st.integers(1, max_rows))
        c = draw(st.integers(1, max_cols))
        return [[draw(cell) for _ in range(c)] for _ in range(r)]

    return build()


def approx_equal(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


# -- P1: plain matrices, optimal vs exhaustive search --------------------------


@settings(max_examples=300, **COMMON)
@given(matrices(ints))
def test_P1_integer_matrices_match_brute_force(matrix):
    result = solve(matrix)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == brute_force_min_cost(matrix)


@settings(max_examples=300, **COMMON)
@given(matrices(floats))
def test_P1_float_matrices_match_brute_force(matrix):
    result = solve(matrix)
    assert_valid_assignment(matrix, result)
    assert approx_equal(total(matrix, result), brute_force_min_cost(matrix))


@settings(max_examples=40, **COMMON)
@given(matrices(ints, max_rows=25, max_cols=25))
def test_P1_larger_matrices_match_scipy(matrix):
    result = solve(matrix, seconds=10)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == scipy_min_cost(matrix)


# -- P2: DISALLOWED. Optimal when feasible, UnsolvableMatrix iff infeasible ----


@pytest.mark.regression
@settings(max_examples=300, phases=[Phase.explicit, Phase.generate], **COMMON)
@given(matrices(ints, forbid=True))
def test_P2_disallowed_feasibility_and_optimality(matrix):
    expected = brute_force_min_cost(matrix)
    if expected is None:
        with pytest.raises(UnsolvableMatrix):
            solve(matrix, seconds=1)
    else:
        result = solve(matrix, seconds=1)
        assert_valid_assignment(matrix, result)
        assert total(matrix, result) == expected


@settings(max_examples=300, **COMMON)
@given(matrices(ints, forbid=True))
def test_P2_disallowed_when_solver_succeeds_it_is_optimal(matrix):
    """Weaker, always-on form: never wrong, even if it can't yet terminate."""
    expected = brute_force_min_cost(matrix)
    if expected is None:
        return  # hang/raise behaviour for infeasible input is covered by B01
    result = solve(matrix, seconds=2)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == expected


# -- P3: input is never mutated -------------------------------------------------


@settings(max_examples=150, **COMMON)
@given(
    matrices(ints),
)
def test_P3_input_not_mutated(matrix):
    snapshot = copy.deepcopy(matrix)
    solve(matrix)
    assert matrix == snapshot


# -- P4: transpose invariance ------------------------------------------------


@settings(max_examples=200, **COMMON)
@given(matrices(ints))
def test_P4_transpose_has_same_optimal_cost(matrix):
    transposed = [list(col) for col in zip(*matrix)]
    a = total(matrix, solve(matrix))
    b = total(transposed, solve(transposed))
    assert a == b


# -- P5: shifting a whole row changes the optimum by exactly that shift ------


@settings(max_examples=200, **COMMON)
@given(
    st.integers(1, 6).flatmap(
        lambda n: st.tuples(
            matrices(ints, max_rows=n, max_cols=n).filter(lambda m: len(m) <= len(m[0])),
            st.integers(-20, 20),
            st.integers(0, 5),
        )
    )
)
def test_P5_row_shift_invariance(args):
    matrix, shift, which = args
    which %= len(matrix)
    shifted = copy.deepcopy(matrix)
    shifted[which] = [v + shift for v in shifted[which]]
    assert total(shifted, solve(shifted)) == total(matrix, solve(matrix)) + shift


# -- P6: maximisation through make_cost_matrix ---------------------------------


@settings(max_examples=200, **COMMON)
@given(matrices(ints))
def test_P6_profit_maximisation(profit):
    cost = make_cost_matrix(profit)
    result = solve(cost)
    assert_valid_assignment(profit, result)
    best = -brute_force_min_cost([[-v for v in row] for row in profit])
    assert total(profit, result) == best


# -- P7: numpy input behaves exactly like list input -------------------------


@pytest.mark.regression
@settings(max_examples=100, **COMMON)
@given(matrices(ints))
def test_P7_numpy_matches_list(matrix):
    np = pytest.importorskip("numpy")
    arr = np.array(matrix)
    assert solve(arr) == solve(matrix)
