"""Metamorphic tests: transformations whose effect on the optimum is known exactly."""

import random
from fractions import Fraction

from munkres import DISALLOWED, solve, solve_planned


def _opt(a, **k):
    return solve(a, **k).total


def test_row_and_column_permutations_preserve_the_optimum():
    rng = random.Random(1)
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        a = [
            [rng.randint(-9, 9) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        try:
            base = _opt(a)
        except Exception:
            continue
        pr, pc = rng.sample(range(n), n), rng.sample(range(m), m)
        b = [[a[i][j] for j in pc] for i in pr]
        assert _opt(b) == base


def test_transpose_preserves_the_optimum():
    rng = random.Random(2)
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        a = [[rng.randint(-9, 9) for _ in range(m)] for _ in range(n)]
        assert _opt(a) == _opt([list(c) for c in zip(*a)])


def test_adding_row_and_column_potentials_shifts_the_objective_exactly():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(1, 6)
        m = rng.randint(n, 6)  # square or wide: every row is used, so row constants always apply
        a = [[rng.randint(-9, 9) for _ in range(m)] for _ in range(n)]
        u = [rng.randint(-5, 5) for _ in range(n)]
        b = [[a[i][j] + u[i] for j in range(m)] for i in range(n)]
        assert _opt(b) == _opt(a) + sum(u)
        if n == m:
            v = [rng.randint(-5, 5) for _ in range(m)]
            c = [[a[i][j] + u[i] + v[j] for j in range(m)] for i in range(n)]
            assert _opt(c) == _opt(a) + sum(u) + sum(v)


def test_positive_scaling_scales_the_objective_and_negation_swaps_min_max():
    rng = random.Random(4)
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        a = [[Fraction(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(m)] for _ in range(n)]
        k = Fraction(rng.randint(1, 7), rng.randint(1, 3))
        assert _opt([[k * v for v in r] for r in a]) == k * _opt(a)
        assert _opt(a, maximize=True) == -_opt([[-v for v in r] for r in a])


def test_planner_matches_plain_solver_on_all_these_transformations():
    rng = random.Random(5)
    for _ in range(100):
        a = [[rng.randint(-9, 9) for _ in range(5)] for _ in range(5)]
        assert solve_planned(a, verify=True).total == _opt(a)
