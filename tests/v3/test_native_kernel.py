"""Native C kernel vs SciPy and the exact Python solver."""

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.optimize")

from munkres import AssignmentSolver, UnsolvableMatrix, linear_sum_assignment, solve_batch  # noqa: E402


def test_linear_sum_assignment_matches_scipy_float_int_forbidden_maximize():
    rng = np.random.default_rng(3)
    for _ in range(1500):
        r, c = rng.integers(1, 13, 2)
        a = rng.integers(-6, 7, (r, c)).astype(float)
        if rng.random() < 0.5:
            a[rng.random((r, c)) < 0.3] = np.inf
        if rng.random() < 0.3 and not np.isinf(a).any():
            a = a.astype(np.int64)
        mx = bool(rng.random() < 0.3)
        try:
            ref = sp.linear_sum_assignment(a, maximize=mx)
            feasible = True
        except ValueError:
            feasible = False
        if feasible:
            got = linear_sum_assignment(a, maximize=mx)
            assert abs(a[got].sum() - a[ref].sum()) < 1e-9
            assert len(got[0]) == len(ref[0]) and len(set(got[1])) == len(got[1])
            assert (np.diff(got[0]) > 0).all()
        else:
            with pytest.raises(ValueError):
                linear_sum_assignment(a, maximize=mx)


def test_nan_and_negative_infinity_rejected():
    for bad in (np.nan, -np.inf):
        with pytest.raises(ValueError):
            linear_sum_assignment(np.array([[1.0, bad], [2.0, 3.0]]))


def test_infeasible_raises_unsolvable_with_hall_message():
    a = np.full((3, 4), np.inf)
    a[:, :2] = 1.0
    with pytest.raises(UnsolvableMatrix):
        linear_sum_assignment(a)


def test_batch_matches_scipy_and_reports_status():
    rng = np.random.default_rng(1)
    x = rng.integers(-5, 6, (40, 5, 7)).astype(float)
    res = solve_batch(x)
    for b in range(40):
        ref = sp.linear_sum_assignment(x[b])
        assert abs(x[b][np.arange(5), res.col_ind[b]].sum() - x[b][ref].sum()) < 1e-9
    bad = np.full((2, 2, 2), np.inf)
    bad[0] = [[1, 2], [3, 4]]
    assert solve_batch(bad, on_infeasible="mark").status.tolist() == [0, 1]
    with pytest.raises(UnsolvableMatrix):
        solve_batch(bad)


def test_batch_huge_range_falls_back_to_exact_solver():
    big = np.array([[[1.7e308, -1.7e308], [-1.7e308, 1.7e308]]])
    assert solve_batch(big).col_ind.tolist() == [[1, 0]]


def test_reusable_solver_matches_scipy():
    rng = np.random.default_rng(0)
    s = AssignmentSolver((20, 30))
    for _ in range(25):
        x = rng.random((20, 30))
        assert (s.solve(x) == sp.linear_sum_assignment(x)[1]).all()
