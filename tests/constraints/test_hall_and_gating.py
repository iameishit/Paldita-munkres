"""Forbidden cells (Hall's condition) and gating, against the independent reference solver."""

import random

import pytest

from munkres import DISALLOWED as D
from munkres import UnsolvableMatrix, diagnose, solve

from ..reference.munkres_reference import min_cost_assignment


def random_matrix(rng, rows, cols, forbid):
    return [
        [D if rng.random() < forbid else rng.randint(-9, 9) for _ in range(cols)]
        for _ in range(rows)
    ]


def test_feasibility_and_optimum_match_the_reference_everywhere():
    rng = random.Random(5)
    for _ in range(600):
        m = random_matrix(rng, rng.randint(1, 6), rng.randint(1, 6), rng.choice([0.2, 0.5, 0.7]))
        expected = min_cost_assignment(m)
        problem = diagnose(m)
        if expected is None:
            assert problem is not None
            with pytest.raises(UnsolvableMatrix):
                solve(m)
        else:
            assert problem is None and solve(m).total == expected[0]


def test_every_witness_is_a_valid_proof_of_infeasibility():
    rng = random.Random(6)
    seen = 0
    for _ in range(600):
        rows, cols = rng.randint(1, 6), rng.randint(1, 6)
        m = random_matrix(rng, rows, cols, 0.6)
        problem = diagnose(m)
        if problem is None:
            continue
        seen += 1
        small, other = (
            (problem.rows, problem.cols) if rows <= cols else (problem.cols, problem.rows)
        )
        neighbours = {
            j
            for i in small
            for j in range(len(m[0]) if rows <= cols else len(m))
            if (m[i][j] if rows <= cols else m[j][i]) is not D
        }
        assert neighbours <= set(other) and len(small) > len(other)
    assert seen > 50


def test_more_gating_never_matches_fewer_pairs():
    rng = random.Random(9)
    for _ in range(150):
        m = random_matrix(rng, rng.randint(1, 5), rng.randint(1, 5), 0.3)
        counts = [len(solve(m, max_cost=t)) for t in range(-10, 12)]
        assert counts == sorted(counts)


def test_loose_gate_equals_no_gate_when_feasible():
    rng = random.Random(10)
    for _ in range(150):
        m = random_matrix(rng, rng.randint(1, 5), rng.randint(1, 5), 0.2)
        if diagnose(m) is None:
            assert solve(m, max_cost=10**6).total == solve(m).total
            assert len(solve(m, max_cost=10**6)) == min(len(m), len(m[0]))


def test_gating_never_raises_even_when_nothing_is_allowed():
    result = solve([[D, D], [D, D]], max_cost=5)
    assert (
        result.pairs == () and result.unmatched_rows == (0, 1) and result.unmatched_cols == (0, 1)
    )
