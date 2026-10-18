import itertools
import random

import pytest

from munkres import (
    DISALLOWED,
    canonical_optimum,
    dulmage_mendelsohn,
    enumerate_optima,
    face_dimension,
    matching_distance,
    most_different_optimum,
    reduce_costs,
    solve,
    topology,
)

from .helpers import all_optima


def _rand(rng, n, m, hi=3, p=0.15):
    return [
        [rng.randint(0, hi) if rng.random() > p else DISALLOWED for _ in range(m)] for _ in range(n)
    ]


def test_enumerate_optima_equals_brute_force_and_topology_count():
    rng = random.Random(1)
    checked = 0
    for _ in range(800):
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        a, mx = _rand(rng, n, m), rng.random() < 0.3
        ref = all_optima(a, mx)
        if ref is None:
            continue
        got = list(enumerate_optima(a, mx))
        assert len(got) == len(set(got)) == len(ref[1]) == topology(a, maximize=mx).n_optimal
        assert {frozenset(g) for g in got} == set(ref[1])
        checked += 1
    assert checked > 400


def test_enumeration_limit():
    a = [[0] * 6 for _ in range(6)]
    with pytest.raises(RuntimeError):
        list(enumerate_optima(a, limit=100))


def test_canonical_optimum_is_lexicographically_smallest_optimum():
    rng = random.Random(2)
    for _ in range(600):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        a, mx = _rand(rng, n, m), rng.random() < 0.3
        ref = all_optima(a, mx)
        if ref is None:
            continue
        want = min(tuple(sorted(s)) for s in ref[1])
        assert canonical_optimum(a, mx) == want


def test_face_dimension_matches_affine_rank_of_optima():
    assert face_dimension([[1, 2], [3, 4]]) == 1  # 1+4 == 2+3: two optima
    assert face_dimension([[0, 0], [0, 0]]) == 1  # two optima -> a segment
    assert face_dimension([[0, 5], [5, 0]]) == 0  # unique optimum
    assert (
        face_dimension([[0] * 3 for _ in range(3)]) == 4
    )  # Birkhoff polytope of order 3 has dimension (3-1)^2


def test_most_different_optimum_and_distance():
    a = [[0, 0], [0, 0]]
    assert most_different_optimum(a, [(0, 0), (1, 1)]) == ((0, 1), (1, 0))
    assert matching_distance([(0, 0), (1, 1)], [(0, 1), (1, 0)]) == 2


def test_reduce_costs_preserves_the_optimum():
    rng = random.Random(3)
    for _ in range(600):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        a = _rand(rng, n, m, hi=9)
        try:
            base = solve(a).total
        except Exception:
            continue
        r = reduce_costs(a)
        assert solve(r.matrix).total + r.constant == base
        assert all(v is DISALLOWED or v >= 0 for row in r.matrix for v in row)


def test_dulmage_mendelsohn_missable_sets_by_enumeration():
    rng = random.Random(4)
    for _ in range(600):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        a = _rand(rng, n, m, p=0.45)
        allowed = [[a[i][j] is not DISALLOWED for j in range(m)] for i in range(n)]
        best, sols = 0, []
        for k in range(min(n, m) + 1):
            for rs in itertools.combinations(range(n), k):
                for cs in itertools.permutations(range(m), k):
                    if all(allowed[r][c] for r, c in zip(rs, cs)):
                        if k > best:
                            best, sols = k, []
                        if k == best:
                            sols.append(list(zip(rs, cs)))
        dm = dulmage_mendelsohn(a)
        assert dm.matching_size == best
        assert dm.rows_missable == {
            i for i in range(n) if any(i not in {r for r, _ in s} for s in sols)
        }
        assert dm.cols_missable == {
            j for j in range(m) if any(j not in {c for _, c in s} for s in sols)
        }


def test_exchange_cycles_always_give_other_optima_and_cover_nearly_all_optional_cells():
    from munkres import Certificate, UnsolvableMatrix, certify, exchange_cycles

    rng = random.Random(5)
    need_total = covered_total = 0
    for _ in range(600):
        n, m = rng.randint(1, 5), rng.randint(1, 6)
        a = _rand(rng, n, m, hi=2, p=0.1)
        try:
            s = solve(a)
        except UnsolvableMatrix:
            continue
        cycles = exchange_cycles(a)
        for c in cycles:
            pairs = set(s.pairs)
            for kind, cell in c:
                (pairs.discard if kind == "remove" else pairs.add)(cell)
            assert len(pairs) == len(s.pairs)
            assert isinstance(certify(a, sorted(pairs)), Certificate)  # still provably optimal
        need = topology(a).optional - set(s.pairs)
        need_total += len(need)
        covered_total += len(need & {cell for c in cycles for k, cell in c if k == "add"})
    assert (
        covered_total >= 0.98 * need_total
    )  # documented limit: rare two-chain moves are not generated


def test_dynamic_snapshot_and_restore_are_independent_and_verified():
    from munkres import AssignmentState

    st = AssignmentState([[1, 2], [2, 1]])
    st.solve()
    snap = st.snapshot()
    st.update_cost(0, 0, 9)
    assert st.objective == 4 and snap.objective == 2 and snap.verify()
    st.restore(snap)
    assert st.objective == 2 and st.verify()
    st.update_cost(1, 1, 7)
    assert snap.objective == 2  # the snapshot is not affected by later changes
