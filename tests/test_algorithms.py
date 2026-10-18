"""The core algorithm: warm start, augmenting paths, orientation, dual feasibility, determinism.

Everything is compared with the independent subset-DP reference in `tests/reference`.
"""

import random

import pytest

from munkres import DISALLOWED as D
from munkres import Munkres, shadow_prices, solve

from .reference.munkres_reference import min_cost_assignment


def grid(rng, rows, cols, *, low=0, high=20, forbid=0.0):
    return [
        [D if rng.random() < forbid else rng.randint(low, high) for _ in range(cols)]
        for _ in range(rows)
    ]


def test_agrees_with_the_reference_on_every_small_shape():
    rng = random.Random(1)
    for rows in range(1, 8):
        for cols in range(1, 8):
            for _ in range(15):
                m = grid(rng, rows, cols, low=-10)
                assert solve(m).total == min_cost_assignment(m)[0]


def test_warm_start_paths_all_give_optimal_answers():
    rng = random.Random(2)
    cases = {
        "all zeros": [[0] * 6 for _ in range(6)],
        "identity optimum": [[0 if i == j else 9 for j in range(6)] for i in range(6)],
        "constant rows": [[i] * 6 for i in range(6)],
        "many ties": grid(rng, 7, 7, high=2),
        "one cheap column": [[1 if j == 0 else 50 for j in range(6)] for _ in range(6)],
        "anti-diagonal": [[0 if i + j == 5 else 7 for j in range(6)] for i in range(6)],
    }
    for name, m in cases.items():
        assert solve(m).total == min_cost_assignment(m)[0], name


def test_transposing_the_problem_never_changes_the_optimum():
    rng = random.Random(3)
    for _ in range(120):
        m = grid(rng, rng.randint(1, 7), rng.randint(1, 7), low=-9, forbid=0.15)
        t = [list(col) for col in zip(*m, strict=True)]
        a, b = min_cost_assignment(m), min_cost_assignment(t)
        assert (a is None) == (b is None)
        if a is not None:
            assert solve(m).total == solve(t).total == a[0]


def test_pairs_are_valid_sorted_and_deterministic():
    rng = random.Random(4)
    for _ in range(80):
        m = grid(rng, rng.randint(1, 8), rng.randint(1, 8), forbid=0.2)
        if min_cost_assignment(m) is None:
            continue
        first, second = Munkres().compute(m), Munkres().compute(m)
        assert first == second == sorted(first)
        assert (
            len({r for r, _ in first})
            == len({c for _, c in first})
            == len(first)
            == min(len(m), len(m[0]))
        )
        assert all(m[r][c] is not D for r, c in first)


def test_duals_are_feasible_and_tight_on_rectangles_too():
    rng = random.Random(5)
    for rows, cols in ((3, 7), (7, 3), (5, 5), (1, 6), (6, 1)):
        for _ in range(25):
            m = grid(rng, rows, cols, low=-9)
            prices = shadow_prices(m)
            for i in range(rows):
                for j in range(cols):
                    assert m[i][j] >= prices.row_prices[i] + prices.col_prices[j]
            for i, j in solve(m).pairs:
                assert m[i][j] == prices.row_prices[i] + prices.col_prices[j]


def test_work_is_bounded_by_the_number_of_columns_per_row():
    rng = random.Random(6)
    for n in (5, 12, 25):
        m = grid(rng, n, n, high=1000)
        trace = solve(m, trace=True).trace
        searches = sum(1 for e in trace.events if e["event"] == "augment_start")
        steps = sum(1 for e in trace.events if e["event"] == "adjust")
        assert (
            searches <= n and steps <= searches * n
        )  # each search visits each column at most once


@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_identical_rows_still_get_distinct_columns(n):
    result = solve([[5, 3, 8, 1, 9, 2, 7, 4, 6, 0][:n]] * n)
    assert sorted(result.cols) == list(range(n)) or len(set(result.cols)) == n
