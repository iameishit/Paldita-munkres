"""Analysis tools: shadow prices, counterfactual, tolerance, k-best, bottleneck."""

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    bottleneck,
    counterfactual,
    k_best,
    shadow_prices,
    solve,
    tolerance,
)

from .helpers import all_matchings

D = DISALLOWED
COMMON = {
    "deadline": None,
    "database": None,
    "derandomize": True,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}


def matrices(max_rows=4, max_cols=4, *, forbid=False):
    cell = st.one_of(st.integers(-9, 9), st.just(D)) if forbid else st.integers(-9, 9)

    @st.composite
    def build(draw):
        r, c = draw(st.integers(1, max_rows)), draw(st.integers(1, max_cols))
        return [[draw(cell) for _ in range(c)] for _ in range(r)]

    return build()


def feasible(matrix):
    return next(all_matchings(matrix), None) is not None


# --------------------------------------------------------------------------
# shadow prices (LP duality = a checkable proof of optimality)
# --------------------------------------------------------------------------


@settings(max_examples=300, **COMMON)
@given(matrices(forbid=True))
def test_shadow_prices_are_a_valid_optimality_certificate(matrix):
    if not feasible(matrix):
        with pytest.raises(UnsolvableMatrix):
            shadow_prices(matrix)
        return
    prices = shadow_prices(matrix)
    solution = solve(matrix)
    for i, row in enumerate(matrix):
        for j, cost in enumerate(row):
            if cost is not D:
                assert cost >= prices.row_prices[i] + prices.col_prices[j]  # dual feasible
    for i, j in solution.pairs:
        assert matrix[i][j] == prices.row_prices[i] + prices.col_prices[j]  # tight on the answer
    assert prices.total == solution.total == min(c for c, _ in all_matchings(matrix))


def test_shadow_prices_empty_and_shapes():
    assert shadow_prices([]).total == 0
    empty = shadow_prices([[], []])
    assert empty.row_prices == (0, 0) and empty.col_prices == ()
    wide = shadow_prices([[4, 1, 3], [2, 0, 5]])
    assert len(wide.row_prices) == 2 and len(wide.col_prices) == 3 and wide.total == 3
    tall = shadow_prices([[1, 2], [3, 4], [0, 9]])
    assert len(tall.row_prices) == 3 and len(tall.col_prices) == 2 and tall.total == 2


# --------------------------------------------------------------------------
# counterfactual / tolerance
# --------------------------------------------------------------------------


@settings(max_examples=200, **COMMON)
@given(matrices(forbid=True), st.data())
def test_counterfactual_and_tolerance_match_exhaustive_search(matrix, data):
    if not feasible(matrix):
        return
    everything = list(all_matchings(matrix))
    best = min(c for c, _ in everything)
    i = data.draw(st.integers(0, len(matrix) - 1))
    j = data.draw(st.integers(0, len(matrix[0]) - 1))
    if matrix[i][j] is D:
        with pytest.raises(ValueError, match="forbidden"):
            counterfactual(matrix, i, j)
        return
    with_pair = [c for c, p in everything if (i, j) in p]
    if with_pair:
        assert counterfactual(matrix, i, j) == min(with_pair) - best
    else:
        with pytest.raises(UnsolvableMatrix):
            counterfactual(matrix, i, j)
    chosen = solve(matrix).pairs
    if (i, j) in chosen:
        without = [c for c, p in everything if (i, j) not in p]
        want = min(without) - best if without else math.inf
        assert tolerance(matrix, i, j) == want
    else:
        with pytest.raises(ValueError, match="not in the optimal"):
            tolerance(matrix, i, j)


def test_what_if_examples_and_errors():
    m = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    assert counterfactual(m, 0, 1) == 0  # already optimal
    assert counterfactual(m, 0, 0) == 1  # forcing 4 leaves [[0,5],[2,2]]: 4+0+2 = 6 vs 5
    assert tolerance(m, 0, 1) == 1
    assert tolerance([[1, D], [D, 1]], 0, 0) == math.inf  # can never be replaced
    with pytest.raises(ValueError, match="outside"):
        counterfactual(m, 3, 0)
    with pytest.raises(ValueError, match="outside"):
        tolerance(m, 0, -1)


# --------------------------------------------------------------------------
# k-best assignments (Murty)
# --------------------------------------------------------------------------


@settings(max_examples=150, **COMMON)
@given(matrices(forbid=True), st.integers(0, 8), st.booleans())
def test_k_best_matches_the_sorted_list_of_all_matchings(matrix, k, maximize):
    sign = -1 if maximize else 1
    everything = sorted((sign * c, p) for c, p in all_matchings(matrix))
    got = k_best(matrix, k, maximize=maximize)
    assert len(got) == min(k, len(everything))
    assert [a.total * sign for a in got] == [c for c, _ in everything[:k]]
    assert len({a.pairs for a in got}) == len(got)  # all distinct
    valid = {tuple(p) for _, p in everything}
    for a in got:
        assert a.pairs in valid and a.maximize is maximize
        assert a.total == sum(matrix[r][c] for r, c in a.pairs)


def test_k_best_example_and_errors():
    m = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    assert [a.total for a in k_best(m, 6)] == [5, 7, 8, 9, 9, 12][:0] + sorted(
        c for c, _ in all_matchings(m)
    )
    assert k_best(m, 0) == [] and k_best([[1, D], [1, D]], 3) == []
    with pytest.raises(ValueError, match="k must be"):
        k_best(m, -1)


# --------------------------------------------------------------------------
# bottleneck
# --------------------------------------------------------------------------


@settings(max_examples=250, **COMMON)
@given(matrices(forbid=True), st.booleans())
def test_bottleneck_is_lexicographically_optimal(matrix, maximize):
    if not feasible(matrix):
        with pytest.raises(UnsolvableMatrix):
            bottleneck(matrix, maximize=maximize)
        return
    sign = -1 if maximize else 1
    best = min((max(sign * matrix[r][c] for r, c in p), sign * c) for c, p in all_matchings(matrix))
    got = bottleneck(matrix, maximize=maximize)
    worst = max(sign * matrix[r][c] for r, c in got.pairs)
    assert (worst, sign * got.total) == best


def test_bottleneck_examples():
    # min-sum picks (0,0)+(1,1) = 1 + 9 = 10 (worst 9); bottleneck prefers 5+5 (worst 5)
    m = [[1, 5], [5, 9]]
    assert solve(m).total == 10
    assert bottleneck(m).total == 10 and max(m[r][c] for r, c in bottleneck(m).pairs) == 5
    assert bottleneck([[1, 100], [2, 3]]).pairs == ((0, 0), (1, 1))
    assert not bottleneck([])  # empty is fine
    with pytest.raises(UnsolvableMatrix):
        bottleneck([[D, D]])  # nothing allowed at all
    with pytest.raises(UnsolvableMatrix):
        bottleneck([[1, D], [1, D]])  # values exist but no complete assignment
