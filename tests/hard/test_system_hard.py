"""Hard system-level tests: proofs, dynamic state, hostile inputs, pathological runtimes, determinism."""

import json
import os
import random
import tempfile
import time

import pytest

import munkres
from munkres import (
    DISALLOWED,
    AssignmentState,
    Certificate,
    HistoryCache,
    UnsolvableMatrix,
    certify,
    check_equivalent,
    enumerate_optima,
    near_optimal,
    solve,
    solve_batch,
    solve_lazy,
    solve_spec,
    topology,
    tropical_permanent,
    verify_pairs,
)

SCALE = float(os.environ.get("PALDITA_HARD_SCALE", "1.0"))


def _n(x):
    return max(1, int(x * SCALE))


def test_certificates_at_scale_accept_the_optimum_and_reject_every_corruption():
    rng = random.Random(1)
    np = pytest.importorskip("numpy")
    sp = pytest.importorskip("scipy.optimize")
    rejected = accepted_alt = 0
    for _ in range(_n(120)):
        n = rng.randint(8, 22)
        m = rng.randint(n, n + 4)
        a = [
            [rng.randint(0, 30) if rng.random() > 0.3 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        arr = np.array([[np.inf if v is DISALLOWED else v for v in r] for r in a], dtype=float)
        try:
            ri, ci = sp.linear_sum_assignment(arr)
        except ValueError:
            continue
        pairs = list(zip(ri.tolist(), ci.tolist()))
        opt = sum(a[i][j] for i, j in pairs)
        cert = certify(a, pairs)
        assert isinstance(cert, Certificate) and cert.verify(a)
        assert Certificate.from_json(cert.to_json()).verify(a)
        # corruption 1: swap the columns of two rows
        for _ in range(3):
            (i, j), (k, l) = rng.sample(pairs, 2)
            alt = [(x, y) for x, y in pairs if x not in (i, k)] + [(i, l), (k, j)]
            ok_cells = a[i][l] is not DISALLOWED and a[k][j] is not DISALLOWED
            res = verify_pairs(a, alt)
            if ok_cells:
                same = sum(a[x][y] for x, y in alt) == opt
                assert bool(res) == same
                rejected += not same
                accepted_alt += same
            else:
                assert not res
        # corruption 2: drop a pair, duplicate a column, use a forbidden cell, tamper each stored dual
        assert not verify_pairs(a, pairs[:-1])
        assert not verify_pairs(a, pairs[:-1] + [(pairs[-1][0], pairs[0][1])])
        for k in range(len(cert.u)):
            u = list(cert.u)
            u[k] += 1
            assert not Certificate(
                cert.version,
                cert.problem_hash,
                cert.shape,
                cert.maximize,
                cert.pairs,
                cert.objective,
                tuple(u),
                cert.v,
                cert.transposed,
            ).verify(a)
        for k in range(len(cert.v)):
            v = list(cert.v)
            v[k] -= 1
            assert not Certificate(
                cert.version,
                cert.problem_hash,
                cert.shape,
                cert.maximize,
                cert.pairs,
                cert.objective,
                cert.u,
                tuple(v),
                cert.transposed,
            ).verify(a)
    assert rejected > 100


def test_hostile_certificate_json_never_executes_or_crashes_unexpectedly():
    a = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    good = json.loads(certify(a, solve(a).pairs).to_json())
    rng = random.Random(2)
    poison = [
        None,
        "1/0",
        "NaN",
        "inf",
        "x",
        10**400,
        -1,
        [],
        {},
        True,
        "1e999",
        "__import__('os').system('true')",
        3.5,
        "0x10",
    ]
    for _ in range(_n(3000)):
        bad = json.loads(json.dumps(good))
        for _ in range(rng.randint(1, 3)):
            key = rng.choice(list(bad))
            if isinstance(bad[key], list) and bad[key] and rng.random() < 0.7:
                bad[key][rng.randrange(len(bad[key]))] = rng.choice(poison)
            elif rng.random() < 0.2:
                del bad[key]
            else:
                bad[key] = rng.choice(poison)
        text = json.dumps(bad)
        try:
            cert = Certificate.from_json(text)
            res = cert.verify(a)  # must be a clean False / True, not an exception
            assert isinstance(bool(res), bool)
        except ValueError:
            pass  # the only allowed failure type (KeyError / TypeError / ZeroDivisionError would be a bug)


def test_dynamic_state_long_random_walks_stay_optimal_and_provable():
    rng = random.Random(3)
    ops_done = 0
    for trial in range(_n(25)):
        n = rng.randint(3, 12)
        m = n + rng.randint(0, 3)
        a = [
            [rng.randint(-20, 40) if rng.random() > 0.1 else DISALLOWED for _ in range(m)]
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
        snaps = []
        for step in range(_n(150)):
            op = rng.choice(
                ["upd"] * 6 + ["addrow", "delrow", "addcol", "delcol", "snap", "restore", "rewind"]
            )
            if op == "upd":
                for _ in range(rng.randint(1, 4)):
                    i, j = rng.choice(sorted(rows)), rng.choice(sorted(cols))
                    v = rng.randint(-20, 40) if rng.random() > 0.12 else DISALLOWED
                    st.update_cost(i, j, v)
                    mat[(i, j)] = v
            elif op == "addrow" and len(rows) < len(cols):
                r = max(rows) + 1
                cs = {
                    j: (rng.randint(-20, 40) if rng.random() > 0.15 else DISALLOWED) for j in cols
                }
                st.add_row(r, cs)
                rows.add(r)
                mat.update({(r, j): v for j, v in cs.items()})
            elif op == "delrow" and len(rows) > 1:
                r = rng.choice(sorted(rows))
                st.remove_row(r)
                rows.discard(r)
            elif op == "addcol":
                c = max(cols) + 1
                cs = {
                    i: (rng.randint(-20, 40) if rng.random() > 0.15 else DISALLOWED) for i in rows
                }
                st.add_column(c, cs)
                cols.add(c)
                mat.update({(i, c): v for i, v in cs.items()})
            elif op == "delcol" and len(cols) > len(rows):
                c = rng.choice(sorted(cols))
                st.remove_column(c)
                cols.discard(c)
            elif op == "snap":
                snaps.append((st.snapshot(), dict(mat), set(rows), set(cols)))
            elif op == "restore" and snaps:
                snap, mat, rows, cols = (
                    snaps[-1][0],
                    dict(snaps[-1][1]),
                    set(snaps[-1][2]),
                    set(snaps[-1][3]),
                )
                st.restore(snap)
            elif op == "rewind":
                continue  # rewinding changes the mutation history; covered by the dedicated test
            else:
                continue
            dense = [[mat.get((i, j), DISALLOWED) for j in sorted(cols)] for i in sorted(rows)]
            try:
                ref = solve(dense).total
            except UnsolvableMatrix:
                with pytest.raises(UnsolvableMatrix):
                    st.solve()
                break
            assert st.objective == ref, (trial, step, op)
            ops_done += 1
            if ops_done % 7 == 0:
                assert st.verify(), (trial, step)
    assert ops_done > _n(400)


def test_all_solver_routes_agree_on_many_random_instances():
    rng = random.Random(4)
    for _ in range(_n(1200)):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        a = [
            [
                rng.choice([0, 0, 1, 2, 3, -2]) if rng.random() > 0.2 else DISALLOWED
                for _ in range(m)
            ]
            for _ in range(n)
        ]
        try:
            solve(a)
        except UnsolvableMatrix:
            continue
        rep = check_equivalent(a, maximize=rng.random() < 0.3)
        assert rep.agree and rep.verified, rep
        t = topology(a)
        assert t.n_optimal == len(list(enumerate_optima(a))) == len(near_optimal(a, 0))
        assert tropical_permanent(a) == solve(a).total


def test_garbage_specs_only_raise_documented_errors():
    rng = random.Random(5)

    def junk(depth=0):
        r = rng.random()
        if depth > 2 or r < 0.3:
            return rng.choice([None, 0, 1, -3, 2.5, "a", True, float("nan"), 10**30, [], {}])
        if r < 0.65:
            return [junk(depth + 1) for _ in range(rng.randint(0, 3))]
        return {
            rng.choice(["cost", "objective", "forbidden", "verify", "limits", "zzz"]): junk(
                depth + 1
            )
            for _ in range(rng.randint(0, 3))
        }

    allowed = (ValueError, UnsolvableMatrix, munkres.ProblemTooLargeError)
    ok = 0
    for _ in range(_n(6000)):
        spec = junk()
        try:
            out = solve_spec(spec)
            json.dumps(out)
            ok += 1
        except allowed:
            pass
        except TypeError as e:
            raise AssertionError(f"undocumented TypeError for {spec!r}: {e}") from e
    for _ in range(_n(1500)):  # structured near-valid specs
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        spec = {
            "cost": [[rng.choice([1, 2, 3, None, 0.5]) for _ in range(m)] for _ in range(n)],
            "objective": rng.choice(["min", "max"]),
            "verify": rng.random() < 0.5,
            "forbidden": [[rng.randint(0, 5), rng.randint(0, 5)] for _ in range(rng.randint(0, 2))],
        }
        try:
            json.dumps(solve_spec(spec))
            ok += 1
        except allowed:
            pass
    assert ok > 100


def test_command_line_never_tracebacks_on_hostile_files():
    from munkres import cli

    rng = random.Random(6)
    lines = [
        "1,2,3",
        "4,5",
        "a,b",
        "",
        "1;2;3",
        "nan,1",
        "inf,2",
        "-inf,1",
        "1e999,1",
        "1,,2",
        "\x00,1",
        "  7 , 8 ",
        "9",
    ]
    with tempfile.TemporaryDirectory() as d:
        for k in range(_n(300)):
            text = "\n".join(rng.choice(lines) for _ in range(rng.randint(0, 5)))
            path = os.path.join(d, f"m{k}.csv")
            with open(path, "w", encoding="utf-8", errors="ignore") as fh:
                fh.write(text)
            for argv in ([path], [path, "--maximize"], ["inspect", path], ["certify", path]):
                code = cli.main(argv)
                assert code in (0, 1, 2), (argv, text)


def test_pathological_runtimes_stay_bounded():
    np = pytest.importorskip("numpy")
    from munkres import linear_sum_assignment

    rng = np.random.default_rng(7)
    cases = {
        "all equal 1000x1000": np.zeros((1000, 1000)),
        "ties 0..2 1000x1000": rng.integers(0, 3, (1000, 1000)).astype(float),
        "random 1000x1000": rng.random((1000, 1000)),
        "i*j rank-one 800": np.outer(np.arange(800.0), np.arange(800.0)),
    }
    plant = np.full((1500, 1500), np.inf)
    plant[np.arange(1500), rng.permutation(1500)] = rng.random(1500)
    m = rng.random((1500, 1500)) < 0.03
    plant[m] = rng.random(m.sum())
    cases["97% forbidden 1500x1500"] = plant
    for name, a in cases.items():
        t = time.perf_counter()
        r, c = linear_sum_assignment(a)
        dt = time.perf_counter() - t
        assert len(set(c.tolist())) == len(c) == a.shape[0], name
        assert dt < 15.0, f"{name} took {dt:.1f}s"


def test_results_are_deterministic_and_identical_across_entry_points():
    np = pytest.importorskip("numpy")
    from munkres import AssignmentSolver, linear_sum_assignment

    rng = np.random.default_rng(8)
    for kind in range(4):
        a = rng.integers(0, 3, (40, 40)).astype(float) if kind % 2 else rng.random((40, 40))
        first = linear_sum_assignment(a)[1].tolist()
        for _ in range(20):
            assert linear_sum_assignment(a)[1].tolist() == first
        assert AssignmentSolver((40, 40)).solve(a).tolist() == first
        assert solve_batch(a[None]).col_ind[0].tolist() == first
        assert [j for _, j in sorted(solve(a.tolist()).pairs)] == first


def test_lazy_solver_survives_adversarial_candidate_lists():
    rng = random.Random(9)
    for _ in range(_n(300)):
        n = rng.randint(1, 8)
        m = rng.randint(n, 10)
        a = [
            [rng.randint(0, 20) if rng.random() > 0.25 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        try:
            ref = solve(a).total
        except UnsolvableMatrix:
            continue
        style = rng.randint(0, 3)
        cands = {
            0: lambda i: [],
            1: lambda i: [rng.randrange(m)],
            2: lambda i: rng.sample(range(m), rng.randint(0, m)),
            3: lambda i: list(range(m)),
        }[style]
        res = solve_lazy(n, m, lambda i, j: a[i][j], cands)
        assert res.total == ref, style


def test_exact_state_scales_to_a_large_problem():
    rng = random.Random(10)
    n = 70
    a = [[rng.randint(0, 100) for _ in range(n)] for _ in range(n)]
    st = AssignmentState(a)
    st.solve()
    t = time.perf_counter()
    for _ in range(_n(120)):
        i, j, v = rng.randrange(n), rng.randrange(n), rng.randint(0, 100)
        a[i][j] = v
        st.update_cost(i, j, v)
        assert st.objective == solve(a).total
    assert st.verify() and time.perf_counter() - t < 60
    cache = HistoryCache()
    cache.solve(a)
    for _ in range(_n(20)):
        i, j, v = rng.randrange(n), rng.randrange(n), rng.randint(0, 100)
        a[i][j] = v
        out = cache.solve(a)
        assert out["reused"] and out["objective"] == solve(a).total
