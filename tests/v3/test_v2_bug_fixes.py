"""Regression tests for defects found in the v2.0.0 audit ."""

from decimal import Decimal
from fractions import Fraction

import munkres
from munkres import DISALLOWED, make_cost_matrix, solve


def test_float_costs_near_overflow_do_not_raise_false_unsolvable():
    """v2.0.0 raised UnsolvableMatrix: dual potentials overflowed float64 to +/-inf."""
    big = 1.7e308
    assert solve([[big, -big], [-big, big]]).pairs == ((0, 1), (1, 0))
    assert solve([[big, 1.0], [1.0, big]]).pairs == ((0, 1), (1, 0))


def test_overflow_range_matches_exact_rational_optimum():
    import random

    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(2, 5)
        a = [
            [rng.choice([-1, 1]) * rng.uniform(0.5, 1) * 1.7e308 for _ in range(n)]
            for _ in range(n)
        ]
        got = sum(Fraction(a[i][j]) for i, j in solve(a).pairs)
        import itertools

        best = min(
            sum(Fraction(a[i][p[i]]) for i in range(n)) for p in itertools.permutations(range(n))
        )
        assert got == best


def test_make_cost_matrix_accepts_decimal_infinity():
    """v2.0.0 raised decimal.InvalidOperation: only float inf was recognised as forbidden."""
    out = make_cost_matrix([[Decimal(3), Decimal("Infinity")], [Decimal(1), Decimal(2)]])
    assert out[0][1] == Decimal("Infinity")
    assert out[0][0] == Decimal(0) and out[1][0] == Decimal(2)


def test_native_and_python_paths_agree(monkeypatch):
    import random

    from munkres import _native

    rng = random.Random(3)
    for _ in range(200):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        a = [
            [rng.randint(-9, 9) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        try:
            native = solve(a).total
        except munkres.UnsolvableMatrix:
            native = None
        monkeypatch.setenv("PALDITA_NO_NATIVE", "1")
        monkeypatch.setattr(_native, "_tried", False)
        monkeypatch.setattr(_native, "_mod", None)
        try:
            pure = solve(a).total
        except munkres.UnsolvableMatrix:
            pure = None
        monkeypatch.delenv("PALDITA_NO_NATIVE")
        monkeypatch.setattr(_native, "_tried", False)
        assert native == pure
