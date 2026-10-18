"""
Hard tests for the native kernel and the dense front end. Scale with PALDITA_HARD_SCALE (default 1.0).
Everything is seeded; failures print the seed and a minimal description.
"""

import hashlib
import os
import threading
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.optimize")

import munkres  # noqa: E402
from munkres import UnsolvableMatrix, linear_sum_assignment, solve  # noqa: E402

SCALE = float(os.environ.get("PALDITA_HARD_SCALE", "1.0"))


def _n(x):
    return max(1, int(x * SCALE))


def _check(a, maximize=False, rtol=1e-9):
    """Compare Paldita with SciPy on feasibility and objective."""
    try:
        ref = sp.linear_sum_assignment(a, maximize=maximize)
        feasible = True
    except ValueError:
        feasible = False
    if not feasible:
        with pytest.raises((ValueError, UnsolvableMatrix)):
            linear_sum_assignment(a, maximize=maximize)
        return False
    got = linear_sum_assignment(a, maximize=maximize)
    assert len(got[0]) == len(ref[0]) == min(a.shape)
    assert len(set(got[0].tolist())) == len(got[0]) and len(set(got[1].tolist())) == len(got[1])
    assert (np.diff(got[0]) > 0).all()  # rows ascending, SciPy-compatible
    x, y = float(a[got].sum()), float(a[ref].sum())
    assert abs(x - y) <= rtol * max(1.0, abs(y)), (x, y)
    return True


def test_differential_fuzz_against_scipy_20000_cases():
    rng = np.random.default_rng(20260101)
    feasible = infeasible = 0
    for _ in range(_n(20000)):
        r, c = rng.integers(1, 41, 2)
        kind = rng.integers(0, 6)
        if kind == 0:
            a = rng.random((r, c))
        elif kind == 1:
            a = rng.integers(0, 4, (r, c)).astype(float)  # ties everywhere
        elif kind == 2:
            a = rng.integers(-(10**12), 10**12, (r, c))  # large int64
        elif kind == 3:
            a = rng.standard_normal((r, c)) * 10.0 ** rng.integers(
                -8, 8, (r, c)
            )  # wide dynamic range
        elif kind == 4:
            a = rng.integers(0, 3, (r, c)).astype(np.float32)
        else:
            a = rng.integers(-5, 6, (r, c)).astype(float)
        if a.dtype.kind == "f" and rng.random() < 0.5:
            a = a.astype(float)
            a[rng.random((r, c)) < rng.random() * 0.95] = np.inf
        ok = _check(a, maximize=bool(rng.random() < 0.3))
        feasible += ok
        infeasible += not ok
    assert feasible > 2000 and infeasible > 500  # the generator exercises both outcomes


def test_float_results_are_optimal_for_the_exact_rational_problem_even_with_ulp_near_ties():
    rng = np.random.default_rng(7)
    worst = Fraction(0)
    for _ in range(_n(1500)):
        n = int(rng.integers(2, 13))
        base = 1.0 + rng.integers(0, 4, (n, n)) * 2.0**-52  # costs that differ by single ulps
        a = base.astype(float)
        got = linear_sum_assignment(a)
        exact = solve(
            [[Fraction(float(v)) for v in row] for row in a]
        ).total  # exact rational optimum
        have = sum(Fraction(float(a[i, j])) for i, j in zip(*got))
        assert have >= exact
        worst = max(worst, have - exact)
    assert worst <= Fraction(1, 10**12), float(
        worst
    )  # float rounding never costs more than 1e-12 here


def test_adversarial_families():
    # all equal: any permutation is optimal, the result must still be a valid permutation
    a = np.full((200, 200), 7.0)
    r, c = linear_sum_assignment(a)
    assert sorted(c.tolist()) == list(range(200)) and a[r, c].sum() == 1400.0
    # rank-one i*j: the anti-diagonal is optimal (rearrangement inequality)
    n = 300
    a = np.outer(np.arange(n), np.arange(n)).astype(float)
    r, c = linear_sum_assignment(a)
    assert a[r, c].sum() == a[np.arange(n), np.arange(n)[::-1]].sum()
    # Monge (-x_i y_j): identity is optimal
    x = np.sort(np.random.default_rng(1).integers(0, 50, 400))
    a = -np.outer(x, x).astype(float)
    r, c = linear_sum_assignment(a)
    assert a[r, c].sum() == a[np.arange(400), np.arange(400)].sum()
    # 99% forbidden with a planted permutation (feasible by construction)
    rng = np.random.default_rng(2)
    n = 500
    a = np.full((n, n), np.inf)
    perm = rng.permutation(n)
    a[np.arange(n), perm] = rng.random(n) * 100
    mask = rng.random((n, n)) < 0.01
    a[mask] = rng.random(mask.sum()) * 100
    assert _check(a)
    # Hall trap: all rows but one only reach two columns -> infeasible, must be reported (not a wrong answer)
    a = np.full((6, 6), np.inf)
    a[:5, :2] = 1.0
    a[5, :] = 2.0
    with pytest.raises(UnsolvableMatrix):
        linear_sum_assignment(a)
    # denormals and signed zeros
    d = np.full((5, 5), 5e-324)
    d[np.arange(5), np.arange(5)] = 0.0
    d[0, 1] = -0.0
    assert _check(d)
    # extreme magnitudes mixed inside every row
    rng = np.random.default_rng(3)
    a = rng.random((20, 20)) * 10.0 ** rng.integers(-300, 300, (20, 20))
    assert _check(a, rtol=1e-9)


def test_int64_boundary_is_exact_and_beyond_it_falls_back_exactly():
    rng = np.random.default_rng(4)
    for n in (2, 5, 17):
        edge = (1 << 62) // (2 * n + 4) - 1  # just inside the kernel's overflow bound
        for mag in (edge, edge + 3, 2 * edge, (1 << 62), (1 << 63) - 1):
            a = rng.integers(-mag, mag, (n, n), dtype=np.int64, endpoint=True)
            got = linear_sum_assignment(a)
            exact = solve([[int(v) for v in row] for row in a]).total  # Python big ints
            assert sum(int(a[i, j]) for i, j in zip(*got)) == exact


def test_every_input_layout_and_dtype_gives_the_same_answer_and_never_mutates_the_input():
    rng = np.random.default_rng(5)
    base = rng.integers(-50, 50, (24, 24)).astype(float)
    ref = linear_sum_assignment(base)[1].tolist()
    big = rng.integers(-50, 50, (48, 48)).astype(float)
    big[::2, ::2] = base  # a strided view of this is the same matrix
    variants = {
        "fortran": np.asfortranarray(base),
        "strided": big[::2, ::2],
        "transposed_view": np.ascontiguousarray(base.T).T,
        "readonly": np.array(base),
        "big_endian": base.astype(">f8"),
        "float32": base.astype(np.float32),
        "int16": base.astype(np.int16),
        "object_ints": base.astype(int).astype(object),
        "list": base.tolist(),
    }
    variants["readonly"].setflags(write=False)
    for name, v in variants.items():
        before = hashlib.sha256(
            np.asarray(v, dtype=object).tobytes()
            if name == "object_ints"
            else np.ascontiguousarray(v).tobytes()
            if hasattr(v, "dtype")
            else repr(v).encode()
        ).hexdigest()
        got = np.asarray(linear_sum_assignment(v)[1]).tolist()
        after = hashlib.sha256(
            np.asarray(v, dtype=object).tobytes()
            if name == "object_ints"
            else np.ascontiguousarray(v).tobytes()
            if hasattr(v, "dtype")
            else repr(v).encode()
        ).hexdigest()
        assert before == after, f"{name}: input was mutated"
        assert sum(base[i, j] for i, j in enumerate(got)) == sum(
            base[i, j] for i, j in enumerate(ref)
        ), name
    # huge unsigned and bool dtypes
    assert linear_sum_assignment(np.array([[2**64 - 1, 0], [0, 2**64 - 1]], dtype=np.uint64))[
        1
    ].tolist() == [1, 0]
    assert linear_sum_assignment(np.array([[True, False], [False, True]]))[1].tolist() == [1, 0]


def test_non_finite_values_and_degenerate_shapes():
    for dt in (np.float64, np.float32):
        for maximize, bad in ((False, np.nan), (False, -np.inf), (True, np.nan), (True, np.inf)):
            a = np.ones((4, 4), dtype=dt)
            a[2, 3] = bad
            with pytest.raises(ValueError):
                linear_sum_assignment(
                    a, maximize=maximize
                )  # NaN always; -inf when minimising; +inf when maximising
        a = np.ones((4, 4), dtype=dt)
        a[2, 3] = -np.inf  # maximising: -inf simply forbids the cell (SciPy semantics)
        r, c = linear_sum_assignment(a, maximize=True)
        assert (2, 3) not in set(zip(r.tolist(), c.tolist()))
        a[2, 3] = np.inf  # minimising: +inf forbids the cell
        r, c = linear_sum_assignment(a)
        assert (2, 3) not in set(zip(r.tolist(), c.tolist()))
    for shape in ((0, 0), (0, 5), (5, 0)):
        r, c = linear_sum_assignment(np.zeros(shape))
        assert len(r) == len(c) == 0
    assert linear_sum_assignment(np.array([[3.0]]))[1].tolist() == [0]
    assert linear_sum_assignment(np.array([[9.0, 1.0, 5.0]]))[1].tolist() == [1]
    assert linear_sum_assignment(np.array([[9.0], [1.0], [5.0]]))[0].tolist() == [1]
    with pytest.raises(UnsolvableMatrix):
        linear_sum_assignment(np.full((3, 3), np.inf))


def test_native_extension_rejects_misuse_without_crashing():
    mod = munkres._native._load()
    if mod is None:
        pytest.skip("native extension not built")
    n, m = 4, 6
    cost = np.zeros((n, m))
    r2c = np.empty(n, dtype=np.int32)
    hall = np.empty(n + m + 4, dtype=np.int32)
    ok = mod.solve(0, n, m, cost, r2c, None, None, hall)
    assert ok == 0
    bad_calls = [
        (0, n, m, np.zeros(3), r2c, None, None, hall),  # cost buffer too small
        (0, n, m, cost, np.empty(1, np.int32), None, None, hall),  # r2c too small
        (0, n, m, cost, r2c, None, None, np.empty(2, np.int32)),  # hall too small
        (0, n, m, cost, r2c, np.empty(1), None, hall),  # u too small
        (0, 6, 4, cost, r2c, None, None, hall),  # n > m
        (0, -1, 4, cost, r2c, None, None, hall),  # negative size
        (7, n, m, cost, r2c, None, None, hall),  # unknown kind
        (0, n, m, cost.T[::2], r2c, None, None, hall),  # non-contiguous cost
        (
            0,
            n,
            m,
            cost,
            np.empty(n, np.int32)[::-1].copy().view(np.uint8)[:n],
            None,
            None,
            hall,
        ),  # wrong item size
        (0, n, m, "text", r2c, None, None, hall),
        (0, n, m, None, r2c, None, None, hall),
    ]
    for args in bad_calls:
        with pytest.raises((ValueError, TypeError, BufferError)):
            mod.solve(*args)
    ro = np.zeros((n, m))
    ro.setflags(write=False)
    ro_out = np.empty(n, dtype=np.int32)
    ro_out.setflags(write=False)
    with pytest.raises((ValueError, TypeError, BufferError)):
        mod.solve(0, n, m, ro, ro_out, None, None, hall)  # output buffer must be writable
    assert mod.solve(0, n, m, ro, r2c, None, None, hall) == 0  # read-only *input* is fine
    rng = np.random.default_rng(6)
    pool = [
        cost,
        r2c,
        hall,
        np.zeros(2),
        None,
        b"abc",
        bytearray(100),
        np.zeros((3, 3)),
        5,
        "x",
        [1, 2],
    ]
    for _ in range(
        _n(3000)
    ):  # random argument soup: only the documented exception types may escape
        args = [int(rng.integers(-2, 9)), int(rng.integers(-2, 9)), int(rng.integers(-2, 9))] + [
            pool[int(rng.integers(0, len(pool)))] for _ in range(5)
        ]
        try:
            mod.solve(*args)
        except (ValueError, TypeError, BufferError, OverflowError):
            pass


def test_concurrent_solves_are_correct_and_thread_safe():
    rng = np.random.default_rng(8)
    mats = [rng.random((60, 60)) for _ in range(8)]
    expected = [sp.linear_sum_assignment(a)[1].tolist() for a in mats]
    errors = []
    solvers = [munkres.AssignmentSolver((60, 60)) for _ in range(8)]

    def worker(k):
        try:
            for _ in range(_n(150)):
                assert linear_sum_assignment(mats[k])[1].tolist() == expected[k]
                assert solvers[k].solve(mats[k]).tolist() == expected[k]
                assert munkres.solve_batch(mats[k][None]).col_ind[0].tolist() == expected[k]
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


def test_no_memory_growth_over_many_calls():
    import resource

    a = np.random.default_rng(9).random((30, 30))
    b = np.random.default_rng(10).random((64, 8, 8))
    for _ in range(2000):  # warm up allocator pools
        linear_sum_assignment(a)
        munkres.solve_batch(b)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(_n(60000)):
        linear_sum_assignment(a)
    for _ in range(_n(6000)):
        munkres.solve_batch(b)
    grown_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert grown_kb < 20_000, f"peak RSS grew by {grown_kb} KB"


def test_native_and_pure_python_agree_on_tie_heavy_matrices():
    rng = np.random.default_rng(11)
    for _ in range(_n(4000)):
        r, c = rng.integers(1, 15, 2)
        a = rng.integers(0, 3, (r, c)).tolist()
        for i in range(r):
            for j in range(c):
                if rng.random() < 0.25:
                    a[i][j] = munkres.DISALLOWED
        try:
            native = solve(a).total
        except UnsolvableMatrix:
            native = None
        with munkres._native.forced_off():
            try:
                pure = solve(a).total
            except UnsolvableMatrix:
                pure = None
        assert native == pure
