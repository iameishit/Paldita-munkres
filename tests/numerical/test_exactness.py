"""Exact arithmetic, huge values and invariances, against the independent reference solver."""

import random
from decimal import Decimal
from fractions import Fraction

import pytest

from munkres import DISALLOWED, Munkres, solve

from ..reference.munkres_reference import min_cost_assignment


def random_matrix(rng, maker, rows, cols, forbid=0.0):
    return [
        [DISALLOWED if rng.random() < forbid else maker(rng) for _ in range(cols)]
        for _ in range(rows)
    ]


MAKERS = {
    "int": lambda rng: rng.randint(-50, 50),
    "fraction": lambda rng: Fraction(rng.randint(-50, 50), rng.randint(1, 9)),
    "decimal": lambda rng: Decimal(rng.randint(-5000, 5000)) / Decimal(100),
    "huge": lambda rng: rng.randint(-(10**40), 10**40),
    "mixed-scale": lambda rng: rng.randint(0, 9) * 10 ** rng.randint(0, 25),
}


@pytest.mark.parametrize("kind", MAKERS)
def test_exact_types_give_exactly_the_reference_optimum(kind):
    rng = random.Random(kind)
    for _ in range(80):
        r, c = rng.randint(1, 6), rng.randint(1, 6)
        m = random_matrix(rng, MAKERS[kind], r, c, forbid=0.25)
        expected = min_cost_assignment(m)
        if expected is None:
            with pytest.raises(ValueError, match=r"."):
                solve(m)
        else:
            result = solve(m)
            assert result.total == expected[0]
            assert type(result.total) is type(expected[0]) or result.total == 0


def test_floats_agree_with_the_reference_up_to_rounding():
    rng = random.Random(3)
    for _ in range(100):
        m = random_matrix(rng, lambda g: g.uniform(-1e3, 1e3), rng.randint(1, 6), rng.randint(1, 6))
        assert solve(m).total == pytest.approx(min_cost_assignment(m)[0], rel=1e-9, abs=1e-9)


def test_scaling_and_shifting_invariances():
    rng = random.Random(8)
    for _ in range(60):
        n = rng.randint(1, 6)
        m = random_matrix(rng, MAKERS["int"], n, n)
        base = solve(m).total
        assert solve([[3 * v for v in row] for row in m]).total == 3 * base
        assert solve([[v + 7 for v in row] for row in m]).total == base + 7 * n
        assert solve([[-v for v in row] for row in m], maximize=True).total == -base


def test_total_keeps_the_input_type():
    assert isinstance(solve([[Fraction(1, 3)]]).total, Fraction)
    assert isinstance(solve([[Decimal("1.5")]]).total, Decimal)
    assert isinstance(solve([[2]]).total, int)
    assert isinstance(solve([[2.5]]).total, float)


def test_booleans_and_mixed_numeric_types():
    assert Munkres().compute([[True, False], [False, True]]) == [(0, 1), (1, 0)]
    assert solve([[1, 2.5], [Fraction(1, 2), 3]]).total == 3.0
