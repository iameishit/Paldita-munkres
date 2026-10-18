import itertools
import math
import random

import pytest

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    count_matchings,
    gibbs,
    minimax_regret_exact,
    minimize_failing_matrix,
    regret_of,
    solve,
)


def test_regret_equals_maximum_over_all_corner_scenarios():
    rng = random.Random(1)
    for _ in range(40):
        n = 3
        low = [[rng.randint(0, 5) for _ in range(n)] for _ in range(n)]
        high = [[low[i][j] + rng.randint(0, 4) for j in range(n)] for i in range(n)]
        for cols in itertools.permutations(range(n)):
            pairs = list(enumerate(cols))
            worst = None
            for corner in itertools.product([0, 1], repeat=n * n):  # all 2^9 box corners
                s = [
                    [(high if corner[i * n + j] else low)[i][j] for j in range(n)] for i in range(n)
                ]
                reg = sum(s[i][j] for i, j in pairs) - solve(s).total
                worst = reg if worst is None else max(worst, reg)
            r = regret_of(pairs, low, high)
            assert r.value == worst
            assert (
                sum(r.scenario[i][j] for i, j in pairs) - solve(r.scenario).total == r.value
            )  # certificate checks out


def test_minimax_regret_exact_is_minimal_and_degenerate_box_has_zero_regret():
    rng = random.Random(2)
    for _ in range(30):
        n = 3
        low = [[rng.randint(0, 5) for _ in range(n)] for _ in range(n)]
        high = [[low[i][j] + rng.randint(0, 3) for j in range(n)] for i in range(n)]
        pairs, val = minimax_regret_exact(low, high)
        assert val == min(
            regret_of(list(enumerate(c)), low, high).value for c in itertools.permutations(range(n))
        )
        _, val0 = minimax_regret_exact(low, low)
        assert val0 == 0  # no uncertainty -> an optimal assignment has zero regret
    with pytest.raises(ValueError):
        minimax_regret_exact([[0] * 8] * 8, [[1] * 8] * 8)


def _all_injections(a):
    n, m = len(a), len(a[0])
    for cols in itertools.permutations(range(m), n):
        if all(a[i][c] is not DISALLOWED for i, c in enumerate(cols)):
            yield cols


def test_count_matchings_and_gibbs_match_enumeration():
    rng = random.Random(3)
    for _ in range(150):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        a = [
            [rng.randint(0, 4) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        sols = list(_all_injections(a))
        assert count_matchings(a) == len(sols)
        if not sols:
            with pytest.raises(UnsolvableMatrix):
                gibbs(a, 0.7)
            continue
        beta = rng.choice([0.0, 0.5, 1.3])
        g = gibbs(a, beta)
        weights = [math.exp(-beta * sum(a[i][c] for i, c in enumerate(s))) for s in sols]
        z = sum(weights)
        assert math.isclose(g.log_z, math.log(z), rel_tol=1e-9, abs_tol=1e-9)
        for i in range(n):
            for j in range(m):
                want = sum(w for s, w in zip(sols, weights) if s[i] == j) / z
                assert math.isclose(g.marginals[i][j], want, abs_tol=1e-9)
        probs = [w / z for w in weights]
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        assert math.isclose(g.entropy, ent, rel_tol=1e-7, abs_tol=1e-9)


def test_gibbs_concentrates_on_the_optimum_at_large_beta():
    a = [[1, 9, 9], [9, 1, 9], [9, 9, 1]]
    g = gibbs(a, 20.0)
    for i in range(3):
        for j in range(
            3
        ):  # the optimum carries (almost) all the mass, everything else (almost) none
            assert (g.marginals[i][j] > 1 - 1e-6) if i == j else (g.marginals[i][j] < 1e-6)
    assert g.expected_cost == pytest.approx(3.0, abs=1e-6)
    assert g.n_matchings == 6


def test_minimizer_shrinks_to_a_one_minimal_failure():
    def fails(m):
        return len(m) >= 2 and any(v is not DISALLOWED and v > 5 for r in m for v in r)

    start = [[1, 7, 2], [3, 4, 9], [8, 1, 1]]
    out = minimize_failing_matrix(start, fails)
    assert fails(out) and len(out) == 2 and len(out[0]) == 1
    assert sum(1 for r in out for v in r if v is not DISALLOWED and v > 5) == 1
    with pytest.raises(ValueError):
        minimize_failing_matrix([[1]], fails)
