import itertools
import random
from fractions import Fraction

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    enumerate_optima,
    explain_edge,
    gibbs,
    ground_state_marginals,
    hall_violator,
    landscape_report,
    near_optimal,
    optimum_diameter,
    sample_optimum,
    solve,
    topology,
    vertex_indispensability,
)

from .helpers import all_optima


def _rand(rng, n, m, hi=3, p=0.2):
    return [
        [rng.randint(0, hi) if rng.random() > p else DISALLOWED for _ in range(m)] for _ in range(n)
    ]


def test_hall_violator_is_a_minimal_violator_and_none_when_feasible():
    rng = random.Random(1)
    seen_bad = 0
    for _ in range(800):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        a = _rand(rng, n, m, p=0.5)
        v = hall_violator(a)
        try:
            solve(a)
            feasible = True
        except UnsolvableMatrix:
            feasible = False
        assert (v is None) == feasible
        if v:
            seen_bad += 1
            rows, cols = v
            nb = lambda s: {j for i in s for j in range(m) if a[i][j] is not DISALLOWED}
            assert set(cols) == nb(rows) and len(cols) < len(rows)
            for r in rows:  # inclusion-minimal
                rest = set(rows) - {r}
                assert not rest or len(nb(rest)) >= len(rest)
    assert seen_bad > 100


def test_optimum_diameter_and_ground_state_marginals():
    assert optimum_diameter([[0, 5], [5, 0]]) == 0
    assert optimum_diameter([[0, 0], [0, 0]]) == 2
    rng = random.Random(2)
    for _ in range(150):
        n, m = rng.randint(1, 3), rng.randint(1, 4)
        n, m = min(n, m), max(n, m)
        a = _rand(rng, n, m, hi=2, p=0.1)
        if all_optima(a) is None:
            continue
        gm = ground_state_marginals(a)
        g = gibbs(a, 60.0)  # beta -> infinity: uniform over optima
        for (i, j), f in gm.items():
            assert abs(g.marginals[i][j] - float(f)) < 1e-6
        assert sum(1 for i in range(n) for j in range(m) if g.marginals[i][j] > 1e-6) == len(gm)


def test_vertex_indispensability_against_enumeration():
    rng = random.Random(3)
    for _ in range(200):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        a = _rand(rng, n, m, hi=5, p=0.1)
        ref = all_optima(a)
        if ref is None:
            continue
        got = vertex_indispensability(a)
        for i in range(n):
            sub = [r for k, r in enumerate(a) if k != i]
            want = (
                0 - ref[0]
                if not sub or not sub[0]
                else (None if all_optima(sub) is None else all_optima(sub)[0] - ref[0])
            )
            assert got["rows"][i] == want
        for j in range(m):
            sub = [[v for k, v in enumerate(r) if k != j] for r in a]
            want = (
                0 - ref[0]
                if not sub[0]
                else (None if all_optima(sub) is None else all_optima(sub)[0] - ref[0])
            )
            assert got["cols"][j] == want


def test_near_optimal_equals_brute_force_shell():
    rng = random.Random(4)
    for _ in range(300):
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        n, m = min(n, m), max(n, m)
        a = _rand(rng, n, m, hi=4, p=0.1)
        ref = all_optima(a)
        if ref is None:
            continue
        eps = rng.randint(0, 4)
        got = near_optimal(a, eps)
        want = []
        for cols in itertools.permutations(range(m), n):
            if all(a[i][c] is not DISALLOWED for i, c in enumerate(cols)):
                v = sum(a[i][c] for i, c in enumerate(cols))
                if v - ref[0] <= eps:
                    want.append((v - ref[0], tuple(enumerate(cols))))
        assert sorted((Fraction(g), p) for g, p in got) == sorted((Fraction(g), p) for g, p in want)


def test_sample_optimum_is_an_optimum_and_reaches_all():
    rng = random.Random(5)
    a = [[0, 0], [0, 0]]
    seen = {sample_optimum(a, rng) for _ in range(100)}
    assert seen == set(enumerate_optima(a))


def test_explain_edge_agrees_with_topology():
    a = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    t = topology(a)
    for cell in t.reduced_cost:
        text = explain_edge(a, *cell)
        key = (
            "forced"
            if cell in t.forced
            else ("optional" if cell in t.optional else "not in any optimal")
        )
        assert key in text
    assert "forbidden" in explain_edge([[1, DISALLOWED], [2, 3]], 0, 1)


def test_landscape_report_is_consistent():
    r = landscape_report([[0, 0], [0, 0]], beta=1.0)
    assert (
        r["n_optimal"] == 2
        and r["face_dimension"] == 1
        and r["n_feasible_assignments"] == 2
        and not r["unique"]
    )
    assert abs(r["gibbs_entropy"] - 0.6931471805599453) < 1e-9
