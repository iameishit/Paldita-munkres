import itertools
import random
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    classify_cost,
    enumerate_optima,
    hall_violator,
    hall_violators,
    solve,
    solve_group_capacity,
    stable_optimum,
)

from .helpers import all_optima


def test_stable_optimum_maximises_the_minimum_margin_among_optima():
    rng = random.Random(1)
    checked = 0
    for _ in range(150):
        n, m = rng.randint(2, 4), rng.randint(2, 4)
        n, m = min(n, m), max(n, m)
        a = [[rng.randint(0, 2) for _ in range(m)] for _ in range(n)]
        ref = all_optima(a)
        base = ref[0]

        def margin(sol):
            vals = []
            for i, j in sol:
                alt = [r[:] for r in a]
                alt[i][j] = DISALLOWED
                try:
                    vals.append(solve(alt).total - base)
                except UnsolvableMatrix:
                    vals.append(float("inf"))
            return min(vals)

        want = max(margin(s) for s in ref[1])
        sol, mm = stable_optimum(a)
        assert mm == want and margin(sol) == want and frozenset(sol) in ref[1]
        checked += 1
    assert checked == 150


def test_hall_violators_are_all_inclusion_minimal_and_include_the_single_witness():
    rng = random.Random(2)
    seen = 0
    for _ in range(300):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        a = [[1 if rng.random() > 0.55 else DISALLOWED for _ in range(m)] for _ in range(n)]
        nb = [{j for j in range(m) if a[i][j] is not DISALLOWED} for i in range(n)]
        want = []
        for k in range(1, n + 1):
            for s in itertools.combinations(range(n), k):
                if len(set().union(*(nb[i] for i in s))) < k and not any(
                    set(w) <= set(s) for w in want
                ):
                    want.append(s)
        got = hall_violators(a)
        assert [g[0] for g in got] == want
        single = hall_violator(a)
        assert (single is None) == (not want)
        if single:
            seen += 1
            assert single[0] in want
    assert seen > 50


def test_classify_cost_tags_follow_exact_facts():
    assert {
        "dense",
        "square",
        "integer",
        "nonnegative",
        "symmetric",
        "toeplitz",
        "monge",
    } <= classify_cost([[1, 2], [2, 1]])
    t = classify_cost([[1.5, DISALLOWED], [DISALLOWED, -2.0]])
    assert (
        {"sparse", "has_forbidden", "float"} <= t and "nonnegative" not in t and "integer" not in t
    )
    assert "circulant" in classify_cost([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
    assert "rectangular" in classify_cost([[1, 2, 3], [4, 5, 6]])
    assert "rational" in classify_cost(
        [[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]]
    )
    assert "additive" in classify_cost([[1, 2], [3, 4]])


def test_group_capacity_assignment_equals_brute_force():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 4)
        m = rng.randint(n, 6)
        a = [
            [rng.randint(0, 9) if rng.random() > 0.1 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        groups = [rng.randint(0, 2) for _ in range(m)]
        caps = {g: rng.randint(0, 3) for g in range(3)}
        mx = rng.random() < 0.3
        vals = []
        for cols in itertools.permutations(range(m), n):
            if all(a[i][c] is not DISALLOWED for i, c in enumerate(cols)) and all(
                sum(1 for c in cols if groups[c] == g) <= caps[g] for g in caps
            ):
                vals.append(sum(a[i][c] for i, c in enumerate(cols)))
        if not vals:
            with pytest.raises(UnsolvableMatrix):
                solve_group_capacity(a, groups, caps, mx)
            continue
        pairs, total = solve_group_capacity(a, groups, caps, mx)
        assert total == (max(vals) if mx else min(vals))
        assert all(sum(1 for _, c in pairs if groups[c] == g) <= caps[g] for g in caps)


def test_parametric_atlas_cache_strip_additive_and_threshold_fit():
    from munkres import (
        ParametricCache,
        calibrate_native_threshold,
        fit_threshold,
        parametric_assignment,
        parametric_atlas,
        strip_additive,
    )

    rng = random.Random(7)
    for _ in range(25):
        n = 3
        a = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        b1 = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        b2 = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        xs, ys = [0, 1, 2, 3], [0, 1, 2]
        atlas = parametric_atlas(a, b1, b2, xs, ys)
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                best = min(
                    sum(a[i][c] + x * b1[i][c] + y * b2[i][c] for i, c in enumerate(p))
                    for p in itertools.permutations(range(n))
                )
                assert atlas["objective"][xi][yi] == best
                pairs = atlas["regions"][atlas["labels"][xi][yi]]
                assert (
                    sum(a[i][c] + x * b1[i][c] + y * b2[i][c] for i, c in pairs) == best
                )  # label's assignment is optimal there
    cache = ParametricCache()
    for _ in range(20):
        a = [[rng.randint(0, 9) for _ in range(3)] for _ in range(3)]
        b = [[rng.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        cache.get(a, b, 0, 8)
        lo = Fraction(rng.randint(0, 3))
        hi = lo + rng.randint(1, 4)
        got = cache.get(a, b, lo, hi)
        want = parametric_assignment(a, b, lo, hi)
        assert [(p.start, p.end, p.base, p.slope) for p in got] == [
            (p.start, p.end, p.base, p.slope) for p in want
        ]
    assert cache.hits == 20 and cache.misses == 20
    for _ in range(100):
        n = rng.randint(1, 4)
        aa, bb = [rng.randint(-5, 5) for _ in range(n)], [rng.randint(-5, 5) for _ in range(n)]
        planted = [
            [aa[i] + bb[j] + rng.choice([0, 0, rng.randint(0, 6)]) for j in range(n)]
            for i in range(n)
        ]
        s = strip_additive(planted)
        assert solve(planted).total == s["constant"] + solve(s["residual"]).total
        assert {frozenset(o) for o in enumerate_optima(planted)} == {
            frozenset(o) for o in enumerate_optima(s["residual"])
        }
        assert all(
            s["residual"][0][j] == 0 and s["residual"][i][0] == 0
            for i in range(n)
            for j in range(n)
        )
    assert (
        strip_additive([[1, 2], [3, 4]])["is_additive"]
        and not strip_additive([[2, 1], [1, 2]])["is_additive"]
    )
    assert fit_threshold([(4, "python"), (9, "python"), (16, "native"), (25, "native")]) == 12.5
    assert fit_threshold([(4, "native"), (9, "native")]) == 4.0
    cal = calibrate_native_threshold(sizes=(2, 6, 16), repeat=5)
    assert cal["threshold_cells"] is None or cal["threshold_cells"] > 0
