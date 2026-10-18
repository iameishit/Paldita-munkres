import itertools
import random
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    maxplus,
    minplus,
    optimality_interval,
    parametric_assignment,
    rank_one_decomposition,
    solve,
    solve_line_matching,
    solve_rank_one,
    tropical_permanent,
)


def _lines(a, b, n):
    return [
        (
            sum(Fraction(a[i][c]) for i, c in enumerate(p)),
            sum(Fraction(b[i][c]) for i, c in enumerate(p)),
            p,
        )
        for p in itertools.permutations(range(n))
    ]


def test_parametric_envelope_matches_brute_force_everywhere():
    rng = random.Random(1)
    for _ in range(120):
        n = rng.randint(2, 4)
        a = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        b = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
        lo, hi = Fraction(0), Fraction(rng.randint(1, 6))
        pieces = parametric_assignment(a, b, lo, hi)
        assert pieces[0].start == lo and pieces[-1].end == hi
        for p, q in zip(pieces, pieces[1:]):
            assert p.end == q.start and p.slope != q.slope  # genuine breakpoints only
        lines = _lines(a, b, n)
        for p in pieces:
            for lam in (p.start, (p.start + p.end) / 2, p.end):
                best = min(x + lam * y for x, y, _ in lines)
                assert p.base + lam * p.slope == best  # the piece is optimal on its whole interval
        # concavity of the minimum: slopes strictly decrease from piece to piece
        assert all(p.slope > q.slope for p, q in zip(pieces, pieces[1:]))


def test_optimality_interval():
    a, b = [[0, 3], [3, 0]], [[1, 0], [0, 1]]
    pairs = [(0, 0), (1, 1)]
    # diagonal costs 2*lambda, anti-diagonal costs 6: diagonal is optimal exactly for lambda <= 3
    assert optimality_interval(a, b, pairs, 0, 10) == (0, 3)
    assert optimality_interval(a, b, [(0, 1), (1, 0)], 0, 10) == (3, 10)
    assert optimality_interval(a, b, [(0, 1), (1, 0)], 0, 2) is None


def test_minplus_maxplus_and_tropical_permanent():
    assert minplus([[1, 2], [3, 4]], [[0, 1], [1, 0]]) == [[1, 2], [3, 4]]
    assert maxplus([[1, 2], [3, 4]], [[0, 1], [1, 0]]) == [[3, 2], [5, 4]]
    rng = random.Random(2)
    for _ in range(400):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        a = [
            [rng.randint(-5, 9) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        mx = rng.random() < 0.3
        try:
            ref = solve(a, maximize=mx).total
        except Exception:
            assert tropical_permanent(a, mx) is None
            continue
        assert tropical_permanent(a, mx) == ref


def test_rank_one_detection_and_solution():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(1, 5)
        a = [rng.randint(-4, 4) for _ in range(n)]
        b = [rng.randint(-4, 4) for _ in range(n)]
        c = [[x * y for y in b] for x in a]
        assert rank_one_decomposition(c) is not None
        pairs = solve_rank_one(c)
        assert sum(c[i][j] for i, j in pairs) == solve(c).total
    assert rank_one_decomposition([[1, 2], [3, 5]]) is None
    with pytest.raises(ValueError):
        solve_rank_one([[1, 2], [3, 5]])


def test_line_matching_equals_brute_force_for_convex_costs():
    rng = random.Random(4)
    for _ in range(300):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        x = [rng.randint(-9, 9) for _ in range(n)]
        y = [rng.randint(-9, 9) for _ in range(m)]
        p = rng.choice([1, 2, 3])
        pairs, total = solve_line_matching(x, y, p)
        best = min(
            sum(abs(x[i] - y[c]) ** p for i, c in enumerate(cols))
            for cols in itertools.permutations(range(m), n)
        )
        assert total == best and len(pairs) == n
