import itertools
import random

import pytest

from munkres import (
    DISALLOWED,
    algebraic_optimum,
    canonical_optimum,
    cooling_schedule,
    energy_barrier,
    energy_landscape,
    has_perfect_matching_algebraic,
    is_popular,
    k_best_diverse,
    near_optimal,
    permanent,
    permanent_bounds,
    popular_matching,
    solve,
    temperature_path,
    verify_canonical,
)

from .helpers import all_optima


def test_algebraic_matching_existence_and_optimum_agree_with_the_primal_solver():
    rng = random.Random(1)
    feas = 0
    for _ in range(300):
        n = rng.randint(1, 5)
        a = [
            [rng.randint(0, 6) if rng.random() > 0.35 else DISALLOWED for _ in range(n)]
            for _ in range(n)
        ]
        try:
            ref = solve(a).total
        except Exception:
            ref = None
        r = random.Random(rng.random())
        assert has_perfect_matching_algebraic(a, r) == (
            ref is not None
        )  # error probability ~ n / 2^61
        assert algebraic_optimum(a, rng=r) == ref
        if ref is not None:
            feas += 1
            assert algebraic_optimum(a, maximize=True, rng=r) == solve(a, maximize=True).total
    assert feas > 100
    with pytest.raises(ValueError):
        algebraic_optimum([[1, 2], [3, -1]])


def test_permanent_exact_and_bounds_bracket_it():
    assert permanent([[1, 2], [3, 4]]) == 10
    assert permanent([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) == 2
    rng = random.Random(2)
    for _ in range(40):
        n = rng.randint(1, 6)
        a = [[rng.uniform(0.2, 3.0) for _ in range(n)] for _ in range(n)]
        per = float(permanent(a))
        lo, hi = permanent_bounds(a)
        assert lo <= per * (1 + 1e-6) and per <= hi * (1 + 1e-6)


def _all_matchings(prefs):
    for choice in itertools.product(*[[None, *p] for p in prefs]):
        used = [c for c in choice if c is not None]
        if len(used) == len(set(used)):
            yield dict(enumerate(choice))


def test_popular_matching_exists_exactly_when_the_algorithm_says_and_is_popular():
    assert (
        popular_matching([[0, 1, 2], [0, 1, 2], [0, 1, 2]]) is None
    )  # the classic instance with no popular matching
    assert not any(is_popular([[0, 1, 2]] * 3, m) for m in _all_matchings([[0, 1, 2]] * 3))
    rng = random.Random(3)
    exist = noexist = 0
    for _ in range(400):
        n = rng.randint(3, 4)
        posts = rng.randint(2, 3)
        base = rng.sample(range(posts), posts)
        prefs = [
            base[:] if rng.random() < 0.6 else rng.sample(range(posts), rng.randint(1, posts))
            for _ in range(n)
        ]
        got = popular_matching(prefs)
        truth = any(is_popular(prefs, m) for m in _all_matchings(prefs))
        assert (got is not None) == truth
        if got is not None:
            exist += 1
            assert is_popular(prefs, got)
            assert len({p for p in got.values() if p is not None}) == sum(
                p is not None for p in got.values()
            )
        else:
            noexist += 1
    assert exist > 50 and noexist > 50


def test_energy_barrier_equals_threshold_connectivity():
    rng = random.Random(4)
    tested = 0
    for _ in range(60):
        n = rng.randint(2, 3)
        m = rng.randint(n, 4)
        a = [[rng.randint(0, 4) for _ in range(m)] for _ in range(n)]
        land = energy_landscape(a, 3)
        nodes = [(x["gap"], x["pairs"]) for x in land["nodes"]]
        adj = {i: set() for i in range(len(nodes))}
        for i, j in land["edges"]:
            adj[i].add(j)
            adj[j].add(i)
        for _ in range(3):
            x, y = rng.choice(nodes), rng.choice(nodes)
            got = energy_barrier(a, x[1], y[1], 3)
            best = None
            for t in sorted({g for g, _ in nodes}):  # BFS on nodes with gap <= t
                ok = {k for k, (g, _) in enumerate(nodes) if g <= t}
                s, t_ = nodes.index(x), nodes.index(y)
                if s in ok and t_ in ok:
                    seen, stack = {s}, [s]
                    while stack:
                        u = stack.pop()
                        for v in adj[u] & ok:
                            if v not in seen:
                                seen.add(v)
                                stack.append(v)
                    if t_ in seen:
                        best = t
                        break
            assert got == best
            tested += 1
    assert tested > 100


def test_k_best_diverse_is_greedy_in_cost_order_with_the_distance_constraint():
    rng = random.Random(5)
    for _ in range(80):
        n, m = 3, 4
        a = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        out = k_best_diverse(a, 4, 2)
        assert out[0][0] == 0
        for (_, p), (_, q) in itertools.combinations(out, 2):
            assert len(set(p) - set(q)) >= 2
        everything = near_optimal(a, 10**9)
        chosen = []
        for g, p in everything:
            if all(len(set(p) - set(q)) >= 2 for _, q in chosen):
                chosen.append((g, p))
            if len(chosen) == 4:
                break
        assert out == chosen


def test_temperature_path_is_monotone_and_converges_to_the_optimum():
    a = [[1, 4, 6], [3, 2, 5], [7, 3, 1]]
    path = temperature_path(a, cooling_schedule(0.05, 40.0, 12))
    ec = [p["expected_cost"] for p in path]
    en = [p["entropy"] for p in path]
    assert all(x >= y - 1e-9 for x, y in zip(ec, ec[1:])) and all(
        x >= y - 1e-9 for x, y in zip(en, en[1:])
    )
    assert abs(ec[-1] - solve(a).total) < 1e-6 and en[-1] < 1e-6
    with pytest.raises(ValueError):
        cooling_schedule(1, 0.5, 5)


def test_verify_canonical_accepts_only_the_lexicographically_smallest_optimum():
    rng = random.Random(6)
    for _ in range(300):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        a = [
            [rng.randint(0, 2) if rng.random() > 0.1 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        ref = all_optima(a)
        if ref is None:
            continue
        smallest = min(tuple(sorted(s)) for s in ref[1])
        assert verify_canonical(a, canonical_optimum(a)).ok
        for s in ref[1]:
            assert bool(verify_canonical(a, sorted(s))) == (tuple(sorted(s)) == smallest)
