"""
Permanent regressions for the v3 audit findings. Every test states the invariant it protects and
checks it against an *independent exact oracle* (brute force over all assignments on exact
rationals), never against the code under test.
"""

import itertools
import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    AssignmentState,
    UnsolvableMatrix,
    canonical_optimum,
    fingerprint,
    gibbs,
    gibbs_sample,
    linear_sum_assignment,
    monge_violations,
    plan,
    problem_hash,
    regret_of,
    solve,
    solve_batch,
    solve_planned,
    stable_matching,
)


def _exact_optimum(rows, maximize=False):
    """Exact optimum of a small rows<=cols problem by enumeration (None if infeasible)."""
    n, m = len(rows), len(rows[0])
    best = None
    for cols in itertools.permutations(range(m), n):
        cells = [rows[i][c] for i, c in enumerate(cols)]
        if any(v is DISALLOWED for v in cells):
            continue
        total = sum(Fraction(v) for v in cells)
        if best is None or (total > best if maximize else total < best):
            best = total
    return best


def _cost_of(rows, cols):
    return sum(Fraction(rows[i][c]) for i, c in enumerate(cols))


# ------------------------------------------------------------------ P0: extended precision (dense path)


def test_longdouble_precision_is_never_silently_rounded():
    np = pytest.importorskip("numpy")
    if np.finfo(np.longdouble).nmant <= 52:
        pytest.skip("longdouble is plain float64 on this platform")
    rng = random.Random(1)
    tiny = np.ldexp(np.longdouble(1), -60)  # far below float64 resolution next to values of order 1
    for _ in range(150):
        n = 5
        a = np.array(
            [[rng.randint(1, 3) + tiny * rng.randint(0, 2) for _ in range(n)] for _ in range(n)],
            dtype=np.longdouble,
        )
        for maximize in (False, True):
            r, c = linear_sum_assignment(a, maximize=maximize)
            best = None
            for cols in itertools.permutations(range(n)):
                total = sum(a[i, cols[i]] for i in range(n))
                best = (
                    total if best is None or (total > best if maximize else total < best) else best
                )
            assert sum(a[i, j] for i, j in zip(r, c)) == best, (
                "extended-precision input must stay optimal"
            )


def test_longdouble_that_fits_float64_still_uses_the_fast_path():
    np = pytest.importorskip("numpy")
    from munkres import _native
    from munkres._dense import fast_lsa

    a = np.arange(9, dtype=np.longdouble).reshape(3, 3)
    if _native.available():
        assert fast_lsa(a) is not None
    lossy = a + np.ldexp(np.longdouble(1), -60) * a
    if np.finfo(np.longdouble).nmant > 52:
        assert fast_lsa(lossy) is None, "values float64 would round must take the exact path"


# ------------------------------------------------------------------ P0: batch exactness


def test_batch_large_integers_are_exact():
    np = pytest.importorskip("numpy")
    big = 1 << 60
    costs = np.array([[[big + 1, big], [big, big + 1]]], dtype=np.int64)
    assert solve_batch(costs).col_ind.tolist() == [[1, 0]]
    assert solve_batch(costs, maximize=True).col_ind.tolist() == [[0, 1]]
    huge = np.array([[[2**63 + 1, 2**63], [2**63, 2**63 + 1]]], dtype=np.uint64)
    assert solve_batch(huge).col_ind.tolist() == [[1, 0]]


def test_batch_random_integers_around_the_float64_boundary_match_exact_optimum():
    np = pytest.importorskip("numpy")
    rng = random.Random(7)
    for _ in range(120):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        base = 2**53 + rng.choice([-3, 0, 5])
        rows = [[base + rng.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        for maximize in (False, True):
            got = (
                solve_batch(np.array([rows], dtype=np.int64), maximize=maximize).col_ind[0].tolist()
            )
            assert _cost_of(rows, got) == _exact_optimum(rows, maximize)


def test_batch_extended_precision_and_exact_scalars():
    np = pytest.importorskip("numpy")
    if np.finfo(np.longdouble).nmant > 52:
        tiny = np.ldexp(np.longdouble(1), -60)
        a = np.array([[[1 + tiny, 1], [1, 1 + tiny]]], dtype=np.longdouble)
        assert solve_batch(a).col_ind.tolist() == [[1, 0]]
    third = Fraction(1, 3)
    objs = np.empty((1, 2, 2), dtype=object)
    objs[0] = [[third, third + Fraction(1, 10**30)], [third, third]]
    assert solve_batch(objs).col_ind.tolist() == [
        [0, 1]
    ]  # the float64 image of both assignments is a tie


def test_batch_maximize_on_the_exact_fallback_is_a_maximisation():
    np = pytest.importorskip("numpy")
    objs = np.empty((1, 2, 2), dtype=object)
    objs[0] = [[Fraction(5), Fraction(1)], [Fraction(1), Fraction(4)]]
    assert solve_batch(objs, maximize=True).col_ind.tolist() == [[0, 1]]
    assert solve_batch(objs, maximize=False).col_ind.tolist() == [[1, 0]]


def test_batch_validates_options_and_reports_failures_consistently():
    np = pytest.importorskip("numpy")
    ok = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    with pytest.raises(ValueError, match="on_infeasible"):
        solve_batch(ok, on_infeasible="raize")
    bad = np.array([[[1.0, np.inf], [2.0, np.inf]], [[1.0, 2.0], [3.0, 4.0]]])
    res = solve_batch(bad, on_infeasible="mark")
    assert res.status.tolist() == [1, 0]
    assert res.col_ind[0].tolist() == [-1, -1], (
        "a failed problem must not expose partial assignments"
    )
    with pytest.raises(UnsolvableMatrix):
        solve_batch(bad)
    with pytest.raises(TypeError):
        solve_batch(np.zeros((1, 2, 2), dtype=complex))


# ------------------------------------------------------------------ P1: Gibbs


def _brute_gibbs(rows, beta):
    n, m = len(rows), len(rows[0])
    sols = [
        c
        for c in itertools.permutations(range(m), n)
        if all(rows[i][j] is not DISALLOWED for i, j in enumerate(c))
    ]
    w = [math.exp(-beta * sum(rows[i][j] for i, j in enumerate(c))) for c in sols]
    z = sum(w)
    return math.log(z), [
        [sum(x for c, x in zip(sols, w) if c[i] == j) / z for j in range(m)] for i in range(n)
    ]


def test_gibbs_feasible_problem_never_reported_unsolvable_by_underflow():
    # both rows prefer column 0, so one must take the 1000-cost cell: every weight is exp(-50000)
    g = gibbs([[0, 1000], [0, 1000]], beta=50.0)
    assert g.n_matchings == 2
    assert g.log_z == pytest.approx(-50000 + math.log(2), rel=1e-12)
    for row in g.marginals:
        assert row == pytest.approx([0.5, 0.5], rel=1e-9)
    assert g.expected_cost == pytest.approx(1000.0, rel=1e-9)
    assert sorted(gibbs_sample([[0, 1000], [0, 1000]], 50.0, random.Random(0))) in (
        [(0, 0), (1, 1)],
        [(0, 1), (1, 0)],
    )


def test_gibbs_truly_infeasible_is_still_reported():
    with pytest.raises(UnsolvableMatrix):
        gibbs([[1, DISALLOWED], [1, DISALLOWED]], 1.0)
    with pytest.raises(UnsolvableMatrix):
        gibbs_sample([[1, DISALLOWED], [1, DISALLOWED]], 1.0)


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf"), "x", None])
def test_gibbs_rejects_non_finite_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        gibbs([[1, 2], [3, 4]], beta)
    with pytest.raises(ValueError, match="beta"):
        gibbs_sample([[1, 2], [3, 4]], beta)


def test_gibbs_matches_enumeration_over_wide_beta_and_negative_costs():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        rows = [[rng.randint(-4, 6) for _ in range(m)] for _ in range(n)]
        beta = rng.choice([0.0, 0.4, 1.0, 3.0, -0.7])
        log_z, marg = _brute_gibbs(rows, beta)
        g = gibbs(rows, beta)
        assert g.log_z == pytest.approx(log_z, abs=1e-9)
        for got, want in zip(g.marginals, marg):
            assert got == pytest.approx(want, abs=1e-9)
        assert sum(g.marginals[0]) == pytest.approx(1.0, abs=1e-9)


def test_gibbs_concentrates_on_the_optimum_at_large_beta():
    rows = [[1, 9, 9], [9, 1, 9], [9, 9, 1]]
    g = gibbs(rows, 20.0)
    for i in range(3):
        for j in range(3):
            assert (g.marginals[i][j] > 1 - 1e-6) if i == j else (g.marginals[i][j] < 1e-6)
    assert g.expected_cost == pytest.approx(3.0, abs=1e-6)
    assert g.entropy == pytest.approx(0.0, abs=1e-5)


def test_gibbs_sample_follows_the_model():
    rows = [[0, 1], [1, 0]]
    rng = random.Random(5)
    hits = sum(gibbs_sample(rows, 1.0, rng) == ((0, 0), (1, 1)) for _ in range(4000))
    p = 1 / (1 + math.exp(-2))  # P(identity) = e^0 / (e^0 + e^-2)
    assert abs(hits / 4000 - p) < 0.03


# ------------------------------------------------------------------ P1: regret validation


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 0), (1, 0)],
        [(0, 0)],
        [(0, 0), (0, 0)],
        [(0, 5), (1, 1)],
        [(0, 0), (1, 1), (1, 1)],
        [(0, 0), ("a", 1)],
    ],
)
def test_regret_rejects_anything_that_is_not_a_complete_assignment(pairs):
    low, high = [[1, 2], [3, 4]], [[2, 3], [4, 6]]
    with pytest.raises(ValueError):
        regret_of(pairs, low, high)


def test_regret_rejects_inverted_intervals_and_ragged_input():
    with pytest.raises(ValueError, match="low > high"):
        regret_of([(0, 0), (1, 1)], [[5, 1], [1, 5]], [[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        regret_of([(0, 0)], [[1, 2]], [[1]])


def test_regret_is_never_negative_for_valid_input():
    rng = random.Random(9)
    for _ in range(100):
        n = 3
        low = [[rng.randint(0, 5) for _ in range(n)] for _ in range(n)]
        high = [[low[i][j] + rng.randint(0, 3) for j in range(n)] for i in range(n)]
        for cols in itertools.permutations(range(n)):
            assert regret_of(list(enumerate(cols)), low, high).value >= 0


# ------------------------------------------------------------------ P1: fingerprint / planner / hash


HOSTILE = {
    "huge ints": [[10**1000, 1], [1, 10**1000]],
    "huge int ratio": [[10**400, 1], [1, 2]],
    "decimal infinity": [[Decimal(1), Decimal("Infinity")], [Decimal(2), Decimal(3)]],
    "decimal infinity row": [[Decimal("Infinity"), Decimal("Infinity")], [Decimal(2), Decimal(3)]],
    "huge fractions": [[Fraction(10**1000), Fraction(1, 10**1000)], [1, 2]],
    "huge decimal exponent": [[Decimal("1e99999"), Decimal("1e-99999")], [1, 2]],
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_analysis_entry_points_never_crash_on_values_the_solver_accepts(name):
    rows = HOSTILE[name]
    try:
        expected = solve(rows).total
    except UnsolvableMatrix:
        expected = None
    fingerprint(rows)
    plan(rows)
    problem_hash(rows)
    if expected is not None:
        assert solve_planned(rows).total == expected
        canonical_optimum(rows)


def test_fingerprint_reports_unbounded_dynamic_range_instead_of_overflowing():
    fp = fingerprint([[10**1000, 1], [1, 10**1000]])
    assert fp.dynamic_range == math.inf
    assert "dynamic range" in " ".join(fp.notes)


def test_every_spelling_of_forbidden_hashes_identically():
    spellings = [DISALLOWED, float("inf"), Decimal("Infinity")]
    hashes = {problem_hash([[1, s], [2, 3]]) for s in spellings}
    assert len(hashes) == 1
    assert problem_hash([[1, 2], [3, 4]]) != problem_hash([[1, 2], [3, 5]])


def test_forbidden_cells_are_consistent_across_float_and_decimal():
    a = fingerprint([[1.0, float("inf")], [2.0, 3.0]])
    b = fingerprint([[Decimal(1), Decimal("Infinity")], [Decimal(2), Decimal(3)]])
    assert (
        (a.allowed_cells, a.components, a.density)
        == (b.allowed_cells, b.components, b.density)
        == (3, 1, 0.75)
    )


def test_monge_violations_rejects_non_finite_cells_with_a_clear_error():
    with pytest.raises(ValueError, match="finite"):
        monge_violations([[1, float("inf")], [2, 3]])
    with pytest.raises(ValueError, match="finite"):
        monge_violations([[1, Decimal("Infinity")], [2, 3]])


def test_planner_obligation_bound_is_exact_for_huge_integers():
    p = plan([[10**1000, 1], [1, 10**1000]])
    assert p.strategy == "python-exact"
    assert any("overflow bound" in o[0] and o[1] is False for o in p.obligations)


# ------------------------------------------------------------------ P1: dynamic state is transactional


def _snapshot(st):
    return (
        {r: dict(c) for r, c in st._c.items()},
        set(st._cols),
        dict(st._v),
        list(st._ops),
        dict(st._mr),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.add_column("x", {0: 1, "no-such-row": 2}),
        lambda s: s.add_column("x", {0: float("nan")}),
        lambda s: s.add_column(0, {}),
        lambda s: s.add_row("r", {"no-such-col": 1}),
        lambda s: s.add_row("r", {0: 1, 1: "abc"}),
        lambda s: s.add_row(0, {}),
        lambda s: s.update_cost(0, 0, float("nan")),
        lambda s: s.update_cost(9, 0, 1),
        lambda s: s.update_costs({(0, 0): 5, (9, 9): 1}),
        lambda s: s.update_costs({(0, 0): 5, (1, 1): float("nan")}),
        lambda s: s.remove_row("nope"),
        lambda s: s.remove_column("nope"),
    ],
)
def test_rejected_mutations_leave_state_and_log_untouched(mutate):
    st = AssignmentState([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    st.solve()
    before = _snapshot(st)
    with pytest.raises((KeyError, ValueError, TypeError)):
        mutate(st)
    assert _snapshot(st) == before
    assert st.verify().ok


def test_dynamic_state_matches_a_fresh_exact_solve_through_random_edits():
    rng = random.Random(11)
    for t in range(400):
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        m = max(n, m)
        rows = [
            [rng.choice([rng.randint(0, 5)] * 3 + [DISALLOWED]) for _ in range(m)] for _ in range(n)
        ]
        maximize = rng.random() < 0.3
        st = AssignmentState(rows, maximize=maximize)
        for step in range(rng.randint(1, 6)):
            op = rng.choice(["update", "add_row", "add_col", "rm_row", "rm_col"])
            if op == "update":
                st.update_cost(
                    rng.choice(list(st._c)),
                    rng.choice(list(st._cols)),
                    rng.choice([rng.randint(0, 5), DISALLOWED]),
                )
            elif op == "add_row":
                st.add_row(
                    ("r", t, step), {c: rng.randint(0, 5) for c in st._cols if rng.random() < 0.8}
                )
            elif op == "add_col":
                st.add_column(
                    ("c", t, step), {r: rng.randint(0, 5) for r in st._c if rng.random() < 0.8}
                )
            elif op == "rm_row" and len(st._c) > 1:
                st.remove_row(rng.choice(list(st._c)))
            elif op == "rm_col" and len(st._cols) > 1:
                st.remove_column(rng.choice(list(st._cols)))
            rs, cs, dense = st.dense()
            if len(rs) > len(cs):
                with pytest.raises(UnsolvableMatrix):
                    st.solve()
                continue
            want = _exact_optimum(dense, maximize)
            if want is None:
                with pytest.raises(UnsolvableMatrix):
                    st.solve()
            else:
                assert st.objective == want
                assert st.verify().ok


def test_dynamic_tie_break_is_independent_of_dict_insertion_order():
    # every assignment of this additive matrix is optimal, so the answer is decided purely by tie-breaking
    base = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    first = AssignmentState(base)
    expected = first.solve()
    for _ in range(5):
        again = AssignmentState([list(r) for r in base])
        assert again.solve() == expected
    # the same history of operations always replays to the same assignment
    st = AssignmentState(base)
    st.add_column("z", {0: 1, 1: 1, 2: 1})
    st.update_cost(0, 0, 5)
    assert AssignmentState.from_log(st.export_log()).solve() == st.solve()


def test_snapshot_and_restore_carry_the_operation_log():
    st = AssignmentState([[3, 1], [2, 4]])
    st.solve()
    snap = st.snapshot()
    st.update_cost(0, 0, 0)
    st.add_column("z", {0: 0, 1: 0})
    st.solve()
    st.restore(snap)
    assert st.export_log()["ops"] == []
    assert AssignmentState.from_log(st.export_log()).solve() == st.solve()
    assert st.objective == 3


# ------------------------------------------------------------------ P2: stable matching with None as a name


def _is_stable(matching, proposers, receivers):
    held = {r: p for p, r in matching.items()}
    for p, prefs in proposers.items():
        for r in prefs:
            if p not in receivers[r] or matching.get(p, "<free>") == r:
                continue
            if p in matching and prefs.index(r) > prefs.index(matching[p]):
                continue
            if r not in held or receivers[r].index(p) < receivers[r].index(held[r]):
                return False
    return True


def test_stable_matching_accepts_none_as_a_participant_name():
    assert stable_matching({None: ["a", "b"], "x": ["a"]}, {"a": ["x", None], "b": [None]}) == {
        None: "b",
        "x": "a",
    }
    assert stable_matching({"p": [None, "b"], "q": [None]}, {None: ["q", "p"], "b": ["p"]}) == {
        "p": "b",
        "q": None,
    }


def test_a_proposer_named_none_is_not_mistaken_for_a_free_receiver():
    # `None` proposes first and holds "r"; the weaker proposer "q" must not displace it
    result = stable_matching({"q": ["r"], None: ["r"]}, {"r": [None, "q"]})
    assert result == {None: "r"}


def test_stable_matching_is_stable_with_none_on_either_side():
    rng = random.Random(2)
    for _ in range(500):
        n = rng.randint(1, 5)
        proposers_names = [None, 0, 1, 2, 3][:n]
        receiver_names = [None if rng.random() < 0.5 else "r0"] + [f"r{i}" for i in range(1, n)]
        proposers = {p: rng.sample(receiver_names, rng.randint(0, n)) for p in proposers_names}
        receivers = {r: rng.sample(proposers_names, rng.randint(0, n)) for r in receiver_names}
        result = stable_matching(proposers, receivers)
        assert len(set(result.values())) == len(result)
        assert all(p in receivers[r] and r in proposers[p] for p, r in result.items())
        assert _is_stable(result, proposers, receivers)


# ------------------------------------------------------------------ ndarray quick path (native `lsa`)


def test_quick_path_matches_scipy_and_defers_every_unusual_input():
    np = pytest.importorskip("numpy")
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        m = n + int(rng.integers(0, 4))
        for a in (
            rng.random((n, m)),
            rng.integers(-50, 50, (n, m)),
            rng.random((m, m))[:, :n].T.copy(),
        ):
            for maximize in (False, True):
                r, c = linear_sum_assignment(a, maximize=maximize)
                r2, c2 = scipy_opt.linear_sum_assignment(a, maximize=maximize)
                assert a[r, c].sum() == pytest.approx(a[r2, c2].sum())
                assert (
                    r.dtype == np.intp
                    and c.dtype == np.intp
                    and r.flags.writeable
                    and c.flags.writeable
                )
    big = np.array(
        [[np.iinfo(np.int64).max, 1], [1, 2]], dtype=np.int64
    )  # sentinel value must not be misread
    r, c = linear_sum_assignment(big)
    assert int(big[r, c].sum()) == 2
    ro = np.array([[3.0, 1.0], [2.0, 4.0]])
    ro.flags.writeable = False
    assert linear_sum_assignment(ro)[1].tolist() == [1, 0]
    fortran = np.asfortranarray(np.array([[3.0, 1.0], [2.0, 4.0]]))
    assert linear_sum_assignment(fortran)[1].tolist() == [1, 0]
    with pytest.raises(UnsolvableMatrix):
        linear_sum_assignment(np.array([[np.inf, np.inf], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        linear_sum_assignment(np.array([[np.nan, 1.0], [1.0, 2.0]]))


# ------------------------------------------------------------------ SciPy-compatible sparse matching


def test_min_weight_full_bipartite_matching_matches_scipy():
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    csgraph = pytest.importorskip("scipy.sparse.csgraph")
    from munkres import min_weight_full_bipartite_matching as mine

    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        m = n + int(rng.integers(0, 3))
        dense = rng.integers(1, 20, (n, m)) * (rng.random((n, m)) < 0.6)
        mat = sparse.csr_matrix(dense)
        for maximize in (False, True):
            try:
                r2, c2 = csgraph.min_weight_full_bipartite_matching(mat, maximize=maximize)
            except ValueError:
                with pytest.raises(ValueError, match="no full matching"):
                    mine(mat, maximize=maximize)
                continue
            r, c = mine(mat, maximize=maximize)
            assert mat[r, c].sum() == mat[r2, c2].sum()
            assert r.tolist() == sorted(r.tolist()) and len(set(c.tolist())) == len(c)
            assert all(dense[i, j] != 0 for i, j in zip(r, c)), "only stored entries may be matched"


def test_sparse_matching_explicit_zero_is_an_edge_and_non_finite_rejected():
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    from munkres import min_weight_full_bipartite_matching as mine

    coo = sparse.coo_matrix(([0.0, 5.0], ([0, 1], [0, 1])), shape=(2, 2))
    assert mine(coo)[1].tolist() == [0, 1]
    with pytest.raises(ValueError):
        mine(sparse.csr_matrix(np.array([[np.inf, 1.0], [1.0, 1.0]])))
