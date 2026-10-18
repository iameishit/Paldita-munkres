import itertools
import math
import os
import random
import tempfile
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    AssignmentState,
    ExperimentRecorder,
    HistoryCache,
    anneal_qubo,
    assignment_to_qubo,
    compact_log,
    decode_bits,
    edge_margins,
    fragility_graph,
    next_breakpoint,
    optimality_length,
    parametric_assignment,
    parametric_degeneracy,
    parametric_duals,
    pareto_assignments,
    qap_branch_and_bound,
    qap_cost,
    qap_gap,
    qubo_energy,
    qubo_to_ising,
    reduce_assignment_problem,
    repair_bits,
    scenario_mass,
    solve,
    solve_lazy,
    solve_memmap,
    solve_points_grid,
    var_assignment,
    verify_candidate,
    warm_start_bits,
)

from .helpers import all_optima


def _bits(k):
    return itertools.product([0, 1], repeat=k)


def test_qubo_minimisers_are_exactly_the_optimal_assignments():
    rng = random.Random(1)
    checked = 0
    for _ in range(60):
        n = rng.randint(1, 3)
        m = rng.randint(n, 3)
        a = [
            [rng.randint(-3, 5) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        ref = all_optima(a)
        if ref is None or n * m > 9:
            continue
        q = assignment_to_qubo(a)
        best = min(qubo_energy(q, b) for b in _bits(len(q.cells)))
        minimisers = {decode_bits(q, b) for b in _bits(len(q.cells)) if qubo_energy(q, b) == best}
        assert None not in minimisers  # the penalty is large enough: no infeasible state ever wins
        assert minimisers == {tuple(sorted(s)) for s in ref[1]}
        assert best == ref[0]
        for b in _bits(len(q.cells)):  # Ising form has identical energies for every state
            s = [2 * x - 1 for x in b]
            ising = qubo_to_ising(q)
            e = (
                ising["const"]
                + sum(h * si for h, si in zip(ising["h"], s))
                + sum(v * s[x] * s[y] for (x, y), v in ising["J"].items())
            )
            assert abs(e - float(qubo_energy(q, b))) < 1e-9
        checked += 1
    assert checked > 30


def test_reduction_warm_start_sampler_repair_and_recorder():
    a = [[1, 9, 9, 9], [9, 1, 9, 9], [9, 9, 1, 9], [2, 2, 2, 0]]
    red = reduce_assignment_problem(a)
    assert red["forced"] and len(red["matrix"]) < len(a)
    assert red["forced_cost"] + solve(red["matrix"]).total == solve(a).total
    assert len(assignment_to_qubo(red["matrix"]).cells) < len(assignment_to_qubo(a).cells)
    q = assignment_to_qubo(a)
    w = warm_start_bits(a, q)
    assert (
        decode_bits(q, w) == tuple(sorted(solve(a).pairs)) and qubo_energy(q, w) == solve(a).total
    )
    hit = 0
    for seed in range(20):
        bits, e = anneal_qubo(q, seed, sweeps=200)
        assert anneal_qubo(q, seed, sweeps=200)[1] == e  # deterministic per seed
        pairs = repair_bits(a, q, bits)
        info = verify_candidate(a, pairs)
        assert info["gap"] >= 0 and len(pairs) == 4 and len({j for _, j in pairs}) == 4
        hit += info["proven_optimal"]
    assert hit >= 1  # a heuristic: at least one of 20 seeded runs reached the proven optimum
    rec = ExperimentRecorder()
    r = rec.record(a, seed=3, sweeps=100)
    assert (
        r["backend"] == "classical-simulated-annealing" and "quantum" not in rec.to_json().lower()
    )


def test_qap_branch_and_bound_is_exact_and_prunes():
    rng = random.Random(2)
    pruned = 0
    for _ in range(40):
        n = rng.randint(3, 6)
        f = [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
        d = [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
        res = qap_branch_and_bound(f, d)
        assert res["cost"] == min(qap_cost(f, d, p) for p in itertools.permutations(range(n)))
        assert qap_cost(f, d, res["permutation"]) == res["cost"]
        pruned += res["nodes"] < res["brute_force_nodes"]
        g = qap_gap(f, d)
        assert g["bound"] <= res["cost"] <= g["incumbent"] and g["gap"] >= 0
    assert pruned > 20


def test_fragility_pareto_var_and_scenario_mass():
    a = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    margins = edge_margins(a)
    for rec in fragility_graph(a):
        assert rec["penalty"] == margins[rec["cell"]]["exclude_penalty"]
        assert rec["cell"] not in rec["alternative"]
    rng = random.Random(3)
    for _ in range(60):
        n = 3
        sc = [
            [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
            for _ in range(rng.randint(2, 4))
        ]
        perms = list(itertools.permutations(range(n)))
        vec = {p: tuple(sum(s[i][c] for i, c in enumerate(p)) for s in sc) for p in perms}
        front = pareto_assignments(sc)
        for p in perms:
            dominated = any(
                q != p and vec[q] != vec[p] and all(x <= y for x, y in zip(vec[q], vec[p]))
                for q in perms
            )
            assert (tuple(enumerate(p)) in front) == (not dominated)
        delta = rng.choice([0.0, 0.25, 0.5])
        k = max(1, math.ceil((1 - delta) * len(sc) - 1e-12))
        assert var_assignment(sc, delta)[1] == min(sorted(v)[k - 1] for v in vec.values())
        mass = scenario_mass(sc)
        for sol, f in mass.items():
            assert f == Fraction(sum(1 for s in sc if frozenset(sol) in all_optima(s)[1]), len(sc))


def test_parametric_extras_match_the_envelope():
    rng = random.Random(4)
    for _ in range(40):
        n = 3
        a = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
        b = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
        pieces = parametric_assignment(a, b, 0, 6)
        lines = [
            (sum(a[i][c] for i, c in enumerate(p)), sum(b[i][c] for i, c in enumerate(p)))
            for p in itertools.permutations(range(n))
        ]
        f = lambda lam: min(x + lam * y for x, y in lines)
        first = pieces[0]
        nb = next_breakpoint(a, b, 0, 6)
        if len(pieces) == 1:
            assert nb is None
        else:
            assert (
                nb == first.end
                and f(nb + Fraction(1, 10**6))
                < first.base + (nb + Fraction(1, 10**6)) * first.slope
            )
            assert f((first.start + nb) / 2) == first.base + (first.start + nb) / 2 * first.slope
        for lam, cnt in parametric_degeneracy(a, b, 0, 6):
            assert cnt >= 2 and cnt == sum(1 for x, y in lines if x + lam * y == f(lam))
        for rec in parametric_duals(a, b, 0, 6):
            assert sum(rec["u"]) + sum(rec["v"]) == sum(
                rec["matrix"][i][j] for i, j in rec["piece"].pairs
            )
        assert optimality_length(a, b, first.pairs, 0, 6) == first.end - first.start


def test_event_log_compaction_rewind_and_history_cache():
    rng = random.Random(5)
    n = 5
    st = AssignmentState([[rng.randint(0, 9) for _ in range(n)] for _ in range(n)])
    st.solve()
    for _ in range(40):
        st.update_cost(rng.randrange(n), rng.randrange(n), rng.randint(0, 9))
    log = st.export_log()
    small = compact_log(log)
    assert len(small["ops"]) < len(log["ops"]) <= 40
    assert AssignmentState.from_log(small).objective == st.objective
    before = st.state_at(20).objective
    st.rewind(20)
    assert st.objective == before and len(st.export_log()["ops"]) == 20 and st.verify()
    cache = HistoryCache()
    base = [[rng.randint(0, 99) for _ in range(30)] for _ in range(30)]
    first = cache.solve(base)
    assert not first["reused"]
    reaug = []
    for _ in range(10):
        nxt = [r[:] for r in base]
        nxt[rng.randrange(30)][rng.randrange(30)] = rng.randint(0, 99)
        out = cache.solve(nxt)
        assert out["reused"] and out["objective"] == solve(nxt).total and out["state"].verify()
        reaug.append(out["reaugmented_rows"])
    assert sum(reaug) / len(reaug) < 5


def test_grid_and_memmap_and_pushdown_lazy_solvers_are_exact():
    rng = random.Random(6)
    for _ in range(15):
        n, m = rng.randint(5, 14), rng.randint(14, 18)
        x = [(rng.random() * 4, rng.random() * 4) for _ in range(n)]
        y = [(rng.random() * 4, rng.random() * 4) for _ in range(m)]
        dense = [[math.dist(p, q) ** 2 for q in y] for p in x]
        res = solve_points_grid(x, y, cell=1.0, k=3)
        ref = solve(dense).total
        assert abs(float(res.total) - float(ref)) <= 1e-9 * max(1, float(ref))
    np = pytest.importorskip("numpy")
    a = np.random.default_rng(1).random((20, 25))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        a.tofile(p)
        res = solve_memmap(p, (20, 25), k=3)
    assert abs(float(res.total) - float(solve(a).total)) < 1e-9
    seen = []
    cost = lambda i, j: (seen.append((i, j)), (i * 3 + j * 5) % 11)[1]
    res = solve_lazy(6, 8, cost, lambda i: [0, 1], allowed=lambda i, j: (i + j) % 3 != 0)
    assert all(
        (i + j) % 3 != 0 for i, j in seen
    )  # the predicate was pushed down: forbidden cells never evaluated
    ref = solve(
        [
            [(i * 3 + j * 5) % 11 if (i + j) % 3 != 0 else DISALLOWED for j in range(8)]
            for i in range(6)
        ]
    ).total
    assert res.total == ref and res.cells_avoided == 6 * 8 - res.evaluations
    rows = lambda i: [(i * 3 + j * 5) % 11 for j in range(8)]
    r2 = solve_lazy(6, 8, lambda i, j: 0, lambda i: [0, 1], cost_rows=rows)
    assert r2.total == solve([[(i * 3 + j * 5) % 11 for j in range(8)] for i in range(6)]).total
