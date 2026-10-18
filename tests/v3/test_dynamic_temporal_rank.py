import itertools
import random

import pytest

from munkres import (
    DISALLOWED,
    AssignmentState,
    UnsolvableMatrix,
    assign_with_inertia,
    rank_maximal,
    solve,
)


def test_dynamic_repair_always_matches_a_fresh_solve_and_is_proven_optimal():
    rng = random.Random(21)
    checked = 0
    for _ in range(150):
        n, m = rng.randint(2, 6), rng.randint(2, 7)
        n, m = min(n, m), max(n, m)
        a = [
            [rng.randint(-9, 9) if rng.random() > 0.15 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        try:
            solve(a)
        except UnsolvableMatrix:
            continue
        st = AssignmentState(a)
        st.solve()
        mat = {(i, j): a[i][j] for i in range(n) for j in range(m)}
        rows, cols = set(range(n)), set(range(m))
        for _ in range(10):
            op = rng.choice(["upd", "upd", "upd", "addrow", "delrow", "addcol", "delcol"])
            if op == "upd":
                for _ in range(rng.randint(1, 3)):
                    i, j = rng.choice(sorted(rows)), rng.choice(sorted(cols))
                    v = rng.randint(-9, 9) if rng.random() > 0.15 else DISALLOWED
                    st.update_cost(i, j, v)
                    mat[(i, j)] = v
            elif op == "addrow" and len(rows) < len(cols):
                r = max(rows) + 1
                cs = {j: (rng.randint(-9, 9) if rng.random() > 0.2 else DISALLOWED) for j in cols}
                st.add_row(r, cs)
                rows.add(r)
                mat.update({(r, j): v for j, v in cs.items()})
            elif op == "delrow" and len(rows) > 1:
                r = rng.choice(sorted(rows))
                st.remove_row(r)
                rows.discard(r)
            elif op == "addcol":
                c = max(cols) + 1
                cs = {i: (rng.randint(-9, 9) if rng.random() > 0.2 else DISALLOWED) for i in rows}
                st.add_column(c, cs)
                cols.add(c)
                mat.update({(i, c): v for i, v in cs.items()})
            elif op == "delcol" and len(cols) > len(rows):
                c = rng.choice(sorted(cols))
                st.remove_column(c)
                cols.discard(c)
            else:
                continue
            dense = [[mat.get((i, j), DISALLOWED) for j in sorted(cols)] for i in sorted(rows)]
            try:
                ref = solve(dense).total
            except UnsolvableMatrix:
                with pytest.raises(UnsolvableMatrix):
                    st.solve()
                break
            assert st.objective == ref
            assert st.verify()
            checked += 1
    assert checked > 500


def test_dynamic_repair_reaugments_few_rows():
    rng = random.Random(3)
    n = 40
    a = [[rng.randint(0, 100) for _ in range(n)] for _ in range(n)]
    st = AssignmentState(a)
    st.solve()
    total = 0
    for _ in range(20):
        st.update_costs({(rng.randrange(n), rng.randrange(n)): rng.randint(0, 100)})
        st.solve()
        total += st.stats.reaugmented_rows
    assert total / 20 < n / 4  # repairs touch a small fraction of rows


def test_inertia_matches_brute_force():
    rng = random.Random(4)
    for _ in range(500):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        n, m = min(n, m), max(n, m)
        a = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
        prev = [rng.choice([None, *range(m)]) for _ in range(n)]
        lam = [rng.randint(0, 6) for _ in range(n)]
        mx = rng.random() < 0.3
        got = assign_with_inertia(a, prev, lam, mx)
        sign = -1 if mx else 1
        vals = [
            sum(
                a[i][c] + sign * lam[i] * (prev[i] is not None and prev[i] != c)
                for i, c in enumerate(cols)
            )
            for cols in itertools.permutations(range(m), n)
        ]
        assert got.objective == (max(vals) if mx else min(vals))


def test_rank_maximal_matches_brute_force():
    rng = random.Random(5)
    for _ in range(400):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        ranks = [[rng.choice([None, 1, 2, 3]) for _ in range(m)] for _ in range(n)]
        cost = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        got = rank_maximal(ranks, cost)
        best = None

        def rec(i, used, pairs):
            nonlocal best
            if i == n:
                sig = [0, 0, 0]
                for x, y in pairs:
                    sig[ranks[x][y] - 1] += 1
                key = (tuple(sig), -sum(cost[x][y] for x, y in pairs))
                best = key if best is None or key > best else best
                return
            rec(i + 1, used, pairs)
            for j in range(m):
                if j not in used and ranks[i][j] is not None:
                    rec(i + 1, used | {j}, [*pairs, (i, j)])

        rec(0, frozenset(), [])
        sig = (*got.signature, *([0] * (3 - len(got.signature))))
        assert (sig, -got.cost) == best


def test_rank_maximal_works_with_ordinal_data_only():
    # idea 119: no cost matrix at all; ties between equal ranks are broken arbitrarily but the signature is optimal
    got = rank_maximal([[1, 2], [1, 2]])
    assert got.signature == (1, 1) and len(got.pairs) == 2
    assert rank_maximal([[1, None], [1, None]]).signature == (
        1,
    )  # only one row can get the shared first choice
