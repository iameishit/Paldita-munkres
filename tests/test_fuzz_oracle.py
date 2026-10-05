"""Seeded, deterministic fuzz against independent oracles. Complements the
Hypothesis tests with the nastier regions: heavy DISALLOWED density, extreme
float magnitudes, and proof-checking of every UnsolvableMatrix witness."""

import random

import pytest

from munkres import DISALLOWED, Munkres, UnsolvableMatrix

from .helpers import assert_valid_assignment, call_with_timeout, scipy_min_cost, total

D = DISALLOWED


def witness_is_valid_proof(matrix, exc):
    """Hall's condition, checked from scratch: |S| > |N(S)| and N(S) within the
    claimed side. Independent of the solver's own bookkeeping."""
    rows, cols = len(matrix), len(matrix[0])
    if rows <= cols:
        s, n = set(exc.rows), set(exc.cols)
        neighbours = {j for i in s for j in range(cols) if matrix[i][j] is not D}
    else:
        s, n = set(exc.cols), set(exc.rows)
        neighbours = {i for j in s for i in range(rows) if matrix[i][j] is not D}
    return bool(s) and neighbours <= n and len(s) > len(n)


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_against_scipy_with_forbidden_cells(seed):
    rng = random.Random(seed)
    for trial in range(1500):
        rows, cols = rng.randint(1, 12), rng.randint(1, 12)
        density = rng.choice([0, 0.2, 0.35, 0.5, 0.7])
        use_float = trial % 3 == 0
        matrix = [
            [
                D
                if rng.random() < density
                else (round(rng.uniform(-30, 60), 3) if use_float else rng.randint(-30, 60))
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]
        expected = scipy_min_cost(matrix)
        try:
            result = call_with_timeout(Munkres().compute, 3, matrix)
        except UnsolvableMatrix as exc:
            assert expected is None, matrix
            assert witness_is_valid_proof(matrix, exc), (matrix, exc.rows, exc.cols)
            continue
        assert expected is not None, "answered an infeasible matrix: %r" % (matrix,)
        assert_valid_assignment(matrix, result)
        assert abs(total(matrix, result) - expected) <= 1e-7 * max(1, abs(expected))


def test_fuzz_extreme_float_magnitudes():
    rng = random.Random(9)
    palette = [1e308, -1e308, 1e300, -1e300, 0.0, 1.0, -1.0, 5e-324, 1e-300, 2.5]
    for _ in range(2000):
        rows, cols = rng.randint(1, 6), rng.randint(1, 6)
        matrix = [[rng.choice(palette) for _ in range(cols)] for _ in range(rows)]
        # no forbidden cells, so it can never be "unsolvable", and never hang
        assert_valid_assignment(matrix, call_with_timeout(Munkres().compute, 3, matrix))
