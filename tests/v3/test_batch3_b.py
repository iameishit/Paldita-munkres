import itertools
import math
import random
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    cvar_assignment,
    mean_variance_assignment,
    scenario_assignment,
    solve,
    solve_3d,
    solve_lazy,
    solve_points,
)


def test_cvar_alpha_extremes_match_expectation_and_worst_case_and_brute_force():
    rng = random.Random(1)
    for _ in range(80):
        n = 3
        sc = [
            [[rng.randint(0, 7) for _ in range(n)] for _ in range(n)]
            for _ in range(rng.randint(2, 5))
        ]
        costs = lambda p: [sum(s[i][c] for i, c in enumerate(p)) for s in sc]
        perms = list(itertools.permutations(range(n)))
        assert (
            cvar_assignment(sc, 0.0)[1] == scenario_assignment(sc)["expected_cost"]
        )  # alpha = 0: the expectation
        assert cvar_assignment(sc, 0.999)[1] == min(
            max(costs(p)) for p in perms
        )  # alpha -> 1: worst case
        alpha = rng.choice([0.25, 0.5, 0.75])
        k = max(1, math.ceil((1 - alpha) * len(sc) - 1e-12))
        want = min(Fraction(sum(sorted(costs(p), reverse=True)[:k]), k) for p in perms)
        assert cvar_assignment(sc, alpha)[1] == want


def test_mean_variance_lambda_zero_is_expected_cost_and_matches_brute_force():
    rng = random.Random(2)
    for _ in range(80):
        n = 3
        sc = [[[rng.randint(0, 7) for _ in range(n)] for _ in range(n)] for _ in range(3)]
        assert mean_variance_assignment(sc, 0)[1] == scenario_assignment(sc)["expected_cost"]
        lam = Fraction(1, 2)

        def val(p):
            cs = [Fraction(sum(s[i][c] for i, c in enumerate(p))) for s in sc]
            mu = sum(cs) / len(cs)
            return mu + lam * sum((c - mu) ** 2 for c in cs) / len(cs)

        assert mean_variance_assignment(sc, lam)[1] == min(
            val(p) for p in itertools.permutations(range(n))
        )
    with pytest.raises(ValueError):
        mean_variance_assignment([[[0] * 9] * 9], 0)


def test_three_index_assignment_equals_brute_force():
    rng = random.Random(3)
    for _ in range(120):
        n = rng.randint(1, 3)
        nj = rng.randint(n, 4)
        nk = rng.randint(n, 4)
        c = [
            [
                [rng.randint(0, 9) if rng.random() > 0.15 else DISALLOWED for _ in range(nk)]
                for _ in range(nj)
            ]
            for _ in range(n)
        ]
        mx = rng.random() < 0.3
        vals = []
        for js in itertools.permutations(range(nj), n):
            for ks in itertools.permutations(range(nk), n):
                cells = [c[i][js[i]][ks[i]] for i in range(n)]
                if all(x is not DISALLOWED for x in cells):
                    vals.append(sum(cells))
        if not vals:
            with pytest.raises(UnsolvableMatrix):
                solve_3d(c, mx)
            continue
        triples, total = solve_3d(c, mx)
        assert total == (max(vals) if mx else min(vals))
        assert sum(c[i][j][k] for i, j, k in triples) == total
        assert len({j for _, j, _ in triples}) == n and len({k for _, _, k in triples}) == n


def test_lazy_solver_is_exact_even_with_bad_candidates_and_uses_few_cells():
    rng = random.Random(4)
    multi_round = 0
    for _ in range(25):
        n = rng.randint(4, 12)
        m = rng.randint(n, 14)
        pts_x = [(rng.random(), rng.random()) for _ in range(n)]
        pts_y = [(rng.random(), rng.random()) for _ in range(m)]
        dense = [[math.dist(p, q) ** 2 for q in pts_y] for p in pts_x]
        ref = Fraction(solve(dense).total)
        k = rng.choice([1, 2, 3, m])
        res = solve_points(pts_x, pts_y, k=k)
        assert res.total == sum(Fraction(dense[i][j]) for i, j in res.pairs)
        assert (
            abs(float(res.total) - float(ref)) <= 1e-9 * max(1.0, float(ref))
            and solve(dense).total <= float(res.total) + 1e-9
        )
        multi_round += res.rounds > 1
        if k < m:
            assert res.candidate_cells < n * m or res.rounds > 1
    assert multi_round > 0  # a poor candidate list was repaired by the exact dual pricing pass


def test_lazy_with_generic_cost_function_and_forbidden_cells():
    rng = random.Random(5)
    for _ in range(60):
        n, m = rng.randint(2, 6), rng.randint(2, 7)
        n = min(n, m)
        a = [
            [rng.randint(0, 9) if rng.random() > 0.3 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        try:
            ref = solve(a).total
        except UnsolvableMatrix:
            with pytest.raises(Exception):
                solve_lazy(n, m, lambda i, j: a[i][j], lambda i: [0], max_rounds=5)
            continue
        res = solve_lazy(n, m, lambda i, j: a[i][j], lambda i: [0])
        assert res.total == ref
