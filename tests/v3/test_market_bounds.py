import itertools
import random
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    AssignmentState,
    CertificateCache,
    Delta,
    UnsolvableMatrix,
    backends,
    check_equivalent,
    gibbs_sample,
    is_competitive_equilibrium,
    market_equilibrium,
    profile_solve,
    prune_dominated_scenarios,
    prune_edges_by_bound,
    qap_cost,
    qap_local_search,
    qap_lower_bound,
    robust_edges,
    scenario_assignment,
    solve,
    solve_with_bound,
    versions,
)

from .helpers import all_optima


def test_market_equilibrium_is_welfare_optimal_and_a_competitive_equilibrium():
    rng = random.Random(1)
    for _ in range(400):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        v = [
            [rng.randint(-2, 9) if rng.random() > 0.15 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        mk = market_equilibrium(v)
        assert is_competitive_equilibrium(v, mk.assignment, mk.prices)
        best = 0

        def rec(i, used, acc):
            nonlocal best
            if i == n:
                best = max(best, acc)
                return
            rec(i + 1, used, acc)
            for j in range(m):
                if j not in used and v[i][j] is not DISALLOWED:
                    rec(i + 1, used | {j}, acc + v[i][j])

        rec(0, frozenset(), 0)
        assert mk.welfare == best
        assert sum(mk.utilities) + sum(mk.prices) == mk.welfare  # strong duality
    # deterministic rejections: a price above the value, and a buyer who would still want the free item
    assert not is_competitive_equilibrium([[5]], {0: 0}, [6])
    assert not is_competitive_equilibrium([[5]], {0: None}, [0])
    assert is_competitive_equilibrium([[5]], {0: 0}, [3])


def test_scenarios_expected_cost_dominance_and_robust_edges():
    rng = random.Random(2)
    for _ in range(150):
        n = 3
        sc = [
            [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
            for _ in range(rng.randint(2, 4))
        ]
        r = scenario_assignment(sc)
        best = min(
            sum(Fraction(sum(s[i][c] for i, c in enumerate(p)), len(sc)) for s in sc)
            for p in itertools.permutations(range(n))
        )
        assert r["expected_cost"] == best
        keep = prune_dominated_scenarios(sc)
        mm = lambda ss: min(
            max(sum(s[i][c] for i, c in enumerate(p)) for s in ss)
            for p in itertools.permutations(range(n))
        )
        assert mm([sc[k] for k in keep]) == mm(sc)
        rb = robust_edges(sc)
        union = lambda s: frozenset.union(*all_optima(s)[1])
        assert rb["possibly_everywhere"] == frozenset.intersection(*[union(s) for s in sc])
        assert rb["always"] == frozenset.intersection(
            *[frozenset.intersection(*all_optima(s)[1]) for s in sc]
        )


def test_bound_aware_solving_and_reduced_cost_fixing_are_sound():
    rng = random.Random(3)
    pruned_any = early = 0
    for _ in range(300):
        n = rng.randint(2, 4)
        m = rng.randint(n, 5)
        a = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
        opt = solve(a).total
        upper = opt + rng.randint(-3, 5)
        r = solve_with_bound(a, upper)
        assert (r.pairs is None) == (opt > upper)
        early += not r.solved
        pm, cnt = prune_edges_by_bound(a, upper) if upper >= opt else (a, 0)
        pruned_any += cnt > 0
        if upper >= opt:
            for cols in itertools.permutations(range(m), n):
                if sum(a[i][c] for i, c in enumerate(cols)) <= upper:
                    assert all(
                        pm[i][c] is not DISALLOWED for i, c in enumerate(cols)
                    )  # nothing useful was removed
            assert solve(pm).total == opt
    assert pruned_any > 20 and early > 5


def test_qap_bound_is_valid_and_local_search_reaches_a_local_optimum():
    rng = random.Random(4)
    for _ in range(80):
        n = rng.randint(3, 6)
        f = [
            [rng.randint(0, 5) if i != j else rng.randint(0, 2) for j in range(n)] for i in range(n)
        ]
        d = [
            [rng.randint(0, 5) if i != j else rng.randint(0, 2) for j in range(n)] for i in range(n)
        ]
        opt = min(qap_cost(f, d, p) for p in itertools.permutations(range(n)))
        assert qap_lower_bound(f, d) <= opt
        start = tuple(rng.sample(range(n), n))
        p, c = qap_local_search(f, d, start)
        assert c <= qap_cost(f, d, start) and c >= opt and c == qap_cost(f, d, p)
        for i in range(n):
            for j in range(i + 1, n):
                q = list(p)
                q[i], q[j] = q[j], q[i]
                assert qap_cost(f, d, q) >= c


def test_gibbs_sampler_matches_exact_probabilities():
    a = [[0, 1, 2], [1, 0, 3]]
    beta = 0.8
    rng = random.Random(7)
    N = 20000
    counts = {}
    for _ in range(N):
        s = gibbs_sample(a, beta, rng)
        counts[s] = counts.get(s, 0) + 1
    import math

    sols = list(itertools.permutations(range(3), 2))
    w = {tuple(enumerate(c)): math.exp(-beta * sum(a[i][j] for i, j in enumerate(c))) for c in sols}
    z = sum(w.values())
    for k, wk in w.items():
        assert abs(counts.get(k, 0) / N - wk / z) < 0.02
    with pytest.raises(UnsolvableMatrix):
        gibbs_sample([[DISALLOWED, DISALLOWED]], 1.0)


def test_certificate_cache_event_log_and_delta():
    a, b = [[1, 2], [2, 1]], [[1, 2], [2, 9]]
    c = CertificateCache()
    r1 = c.verify(a, [(0, 0), (1, 1)])
    r2 = c.verify(a, [(1, 1), (0, 0)])
    assert r1.ok and r2.ok and c.hits == 1 and c.misses == 1
    assert not c.verify(b, [(0, 0), (1, 1)]).ok and c.misses == 2  # cost 10 vs optimal 4
    rng = random.Random(1)
    n = 5
    st = AssignmentState([[rng.randint(0, 9) for _ in range(n)] for _ in range(n)])
    st.solve()
    for _ in range(15):
        st.update_cost(rng.randrange(n), rng.randrange(n), rng.randint(0, 9))
        st.solve()
    st.add_column(99, {r: rng.randint(0, 9) for r in range(n)})
    st.remove_column(2)
    copy = AssignmentState.from_log(st.export_log())
    assert copy.objective == st.objective and copy.verify()
    d1, d2 = Delta({(0, 0): 5, (1, 1): 6}), Delta({(0, 0): 7})
    s1, s2 = AssignmentState([[1, 2], [2, 1]]), AssignmentState([[1, 2], [2, 1]])
    s1.apply(d1)
    s1.apply(d2)
    s2.apply(d1.compose(d2))
    assert s1.objective == s2.objective


def test_backends_versions_profile_and_equivalence():
    names = {b["name"] for b in backends()}
    assert names == {"native-kernel", "python-exact", "components"}
    from fractions import Fraction as F  # noqa: N817

    from munkres import _native

    if _native.available():
        assert (
            _native.solve_dense(2, 2, [[(0, F(1)), (1, F(2))], [(0, F(2)), (1, F(1))]]) is None
        )  # not in its dtype list
    v = versions()
    assert {"package", "api", "semantics", "certificate", "native_abi"} <= set(v)
    pr = profile_solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert (
        pr["verified"]
        and sum(pr["stages_ns"].values()) <= pr["total_ns"]
        and all(x > 0 for x in pr["stages_ns"].values())
    )
    rng = random.Random(9)
    for _ in range(60):
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        n, m = min(n, m), max(n, m)
        a = [[rng.randint(-5, 9) for _ in range(m)] for _ in range(n)]
        rep = check_equivalent(a, maximize=rng.random() < 0.3)
        assert rep.agree and rep.verified, rep
