"""
Regression tests for behaviour fixed in the 2.x series. Each test pins one behaviour that
used to go wrong: hangs on impossible or non-finite input, ragged and empty input, numpy and
tuple input, thread safety, `DISALLOWED` copying and pickling, `make_cost_matrix`, and very
large integers. They run with a hang guard, so a regression fails fast instead of freezing.
"""

import copy
import pickle
import sys
import threading

import pytest

from munkres import DISALLOWED, Munkres, UnsolvableMatrix, _core, make_cost_matrix

from .helpers import assert_valid_assignment, call_with_timeout, scipy_min_cost, total

D = DISALLOWED
INF = float("inf")
NAN = float("nan")


def solve(matrix, seconds=2):
    """Run compute() with a hang guard; fresh solver each time."""
    return call_with_timeout(Munkres().compute, seconds, matrix)


# ---------------------------------------------------------------------------
# B01 -- infeasible DISALLOWED patterns must raise UnsolvableMatrix, not hang
# ---------------------------------------------------------------------------

HALL_VIOLATIONS = [
    # two rows that can only use the same single column
    [[1, D, D], [1, D, D], [1, 2, 3]],
    # same shape, zeros
    [[D, D, 0], [D, D, 0], [0, 0, 0]],
    # three rows competing for two columns
    [[1, 2, D, D], [3, 4, D, D], [5, 6, D, D], [1, 1, 1, 1]],
    # rectangular (wide): rows 0 and 1 share one allowed column
    [[D, D, 5], [D, D, 7]],
    # rectangular (tall): columns 0 and 1 can only use row 2
    [[D, D], [D, D], [4, 9]],
]


@pytest.mark.regression
@pytest.mark.parametrize("matrix", HALL_VIOLATIONS)
def test_B01_infeasible_raises_instead_of_hanging(matrix):
    with pytest.raises(UnsolvableMatrix):
        solve(matrix)


# ---------------------------------------------------------------------------
# B02 / B03 / B04 -- non-finite numbers
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B02_nan_rejected():
    with pytest.raises(ValueError):
        solve([[NAN, 1], [2, 3]])


@pytest.mark.regression
def test_B03_negative_infinity_rejected():
    with pytest.raises(ValueError):
        solve([[-INF, 1], [2, 3]])


@pytest.mark.regression
def test_B04_positive_infinity_row_is_unsolvable():
    with pytest.raises(UnsolvableMatrix):
        solve([[INF, INF], [1, 2]])


@pytest.mark.regression
def test_B04_positive_infinity_acts_as_disallowed():
    # (this already works by accident for easy cases; keep it working)
    result = solve([[INF, 1], [1, INF]])
    assert sorted(result) == [(0, 1), (1, 0)]


# ---------------------------------------------------------------------------
# B05 -- ragged input
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B05_ragged_short_second_row_rejected():
    with pytest.raises(ValueError):
        solve([[1, 2], [3]])


@pytest.mark.regression
def test_B05_ragged_long_later_row_rejected():
    with pytest.raises(ValueError):
        solve([[1, 2], [5, 1, 0, 9], [3, 3, 3, 3], [4, 4, 4, 4]])


# ---------------------------------------------------------------------------
# B06 -- empty input
# Decision: an empty problem has exactly one valid answer (the empty
# assignment), like SciPy. Raising would force callers such as trackers, who
# routinely see frames with zero detections, to special-case it.
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize("matrix", [[], [[]], [[], []], ((),)])
def test_B06_empty_matrix_is_empty_assignment(matrix):
    assert solve(matrix) == []


@pytest.mark.regression
def test_B06_empty_numpy_arrays():
    np_ = pytest.importorskip("numpy")
    assert solve(np_.zeros((0, 3))) == []
    assert solve(np_.zeros((3, 0))) == []


@pytest.mark.regression
def test_B06_ragged_with_empty_row_still_rejected():
    with pytest.raises(ValueError):
        solve([[], [1]])


# ---------------------------------------------------------------------------
# B07 -- tuples
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B07_tuple_matrix_square():
    matrix = ((4, 1, 3), (2, 0, 5), (3, 2, 2))
    result = solve(matrix)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == 5


@pytest.mark.regression
def test_B07_tuple_matrix_rectangular():
    matrix = ((1, 2), (3, 4), (5, 6))
    result = solve(matrix)
    assert_valid_assignment(matrix, result)


# ---------------------------------------------------------------------------
# B08 -- numpy
# ---------------------------------------------------------------------------

np = pytest.importorskip("numpy")


@pytest.mark.regression
@pytest.mark.parametrize("dtype", [int, float])
def test_B08_numpy_input_not_mutated(dtype):
    arr = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=dtype)
    before = arr.copy()
    solve(arr)
    assert (arr == before).all()


@pytest.mark.regression
def test_B08_numpy_rectangular():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    result = solve(arr)
    assert_valid_assignment(arr.tolist(), result)


@pytest.mark.regression
def test_B08_pad_matrix_does_not_mutate_numpy():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    before = arr.copy()
    try:
        Munkres().pad_matrix(arr, 7)
    except Exception:
        pass
    assert (arr == before).all()


# ---------------------------------------------------------------------------
# B09 -- thread safety
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B09_compute_leaves_no_state_on_instance():
    """Deterministic structural check: a call must not leave state behind."""
    solver = Munkres()
    before = dict(vars(solver))
    solver.compute([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert dict(vars(solver)) == before


@pytest.mark.regression
def test_B09_shared_instance_stress():
    """
    Behavioural check (probabilistic before the fix, so not xfail): many
    threads sharing ONE instance must all get optimal answers.
    """
    import random

    rng = random.Random(5)
    mats = [[[rng.randint(0, 50) for _ in range(12)] for _ in range(12)] for _ in range(8)]
    expected = [scipy_min_cost(mx) for mx in mats]
    shared = Munkres()
    failures = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force aggressive thread interleaving
    try:

        def worker(idx):
            for _ in range(25):
                try:
                    res = shared.compute(mats[idx])
                    assert_valid_assignment(mats[idx], res)
                    if total(mats[idx], res) != expected[idx]:
                        failures.append(idx)
                except Exception as exc:
                    failures.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(mats))]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        sys.setswitchinterval(old)
    assert not failures, failures[:3]


# ---------------------------------------------------------------------------
# B10 -- DISALLOWED must survive copy / pickle
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B10_deepcopy_preserves_identity():
    assert copy.deepcopy(DISALLOWED) is DISALLOWED
    assert copy.copy(DISALLOWED) is DISALLOWED


@pytest.mark.regression
def test_B10_pickle_preserves_identity():
    assert pickle.loads(pickle.dumps(DISALLOWED)) is DISALLOWED


@pytest.mark.regression
def test_B10_deepcopied_matrix_still_solvable():
    matrix = [[1, D], [2, 3]]
    result = solve(copy.deepcopy(matrix))
    assert sorted(result) == [(0, 0), (1, 1)]


# ---------------------------------------------------------------------------
# B11 -- make_cost_matrix with DISALLOWED
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B11_make_cost_matrix_keeps_disallowed():
    cost = make_cost_matrix([[5, D], [2, 3]])
    assert cost[0][1] is D
    assert cost[0][0] == 0 and cost[1][0] == 3 and cost[1][1] == 2


# ---------------------------------------------------------------------------
# B12 -- sys.maxsize sentinel
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_B12_huge_integers():
    big = 2**200
    matrix = [[0, big], [0, big]]
    # Both columns can't be used by cost 0 -> optimum is 0 + big.
    result = solve(matrix)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == big


# ---------------------------------------------------------------------------
# Behaviour that must NEVER regress (passes today, must pass after rewrite)
# ---------------------------------------------------------------------------


def test_result_is_sorted_by_row_and_uses_python_ints():
    result = Munkres().compute([[5, 9, 1], [10, 3, 2], [8, 7, 4]])
    assert result == sorted(result)
    assert all(type(r) is int and type(c) is int for r, c in result)


def test_list_input_is_never_mutated():
    matrix = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    snapshot = copy.deepcopy(matrix)
    Munkres().compute(matrix)
    assert matrix == snapshot


@pytest.mark.parametrize(
    "matrix,expected",
    [
        ([[5]], 5),
        ([[5, 3, 9]], 3),  # 1 x N
        ([[5], [3], [9]], 3),  # N x 1
        ([[1, 2, 3], [4, 5, 6]], 6),  # wide
        ([[1, 2], [3, 4], [0, 9]], 2),  # tall
        ([[-1, -5], [-3, -2]], -8),  # negative costs
    ],
)
def test_small_shapes(matrix, expected):
    result = solve(matrix)
    assert_valid_assignment(matrix, result)
    assert total(matrix, result) == expected


def test_all_disallowed_row_is_unsolvable():
    with pytest.raises(UnsolvableMatrix):
        solve([[D, D], [1, 2]])


def test_float_precision_sums():
    matrix = [[0.1 + 0.2, 0.3], [0.3, 0.1 + 0.2]]
    result = solve(matrix)
    assert_valid_assignment(matrix, result)


# ---------------------------------------------------------------------------
# Performance guards. The budgets are ~50x looser than a normal machine needs
# (random 200x200 ~0.09s, tie-heavy ~0.02s), so they only trip on a real
# algorithmic regression -- e.g. the warm start being removed makes the
# tie-heavy case ~25x slower.
# ---------------------------------------------------------------------------


def test_performance_random_200x200():
    import random
    import time

    rng = random.Random(0)
    matrix = [[rng.randint(0, 1000) for _ in range(200)] for _ in range(200)]
    start = time.perf_counter()
    result = Munkres().compute(matrix)
    assert time.perf_counter() - start < 5.0
    assert total(matrix, result) == scipy_min_cost(matrix)


class _CountingList(list):
    """edges list that counts row look-ups == how much solver work was done."""

    reads = 0

    def __getitem__(self, index):
        type(self).reads += 1
        return super().__getitem__(index)


def _solver_row_reads(matrix):
    n, m, edges = _core._validated_edges(matrix)
    _CountingList.reads = 0
    _core._solve(n, m, _CountingList(edges))
    return _CountingList.reads, n


def test_warm_start_resolves_easy_problems_without_any_search():
    """Deterministic (no clocks): if every row has its own zero after row
    reduction the warm start must finish the job -- exactly one look-up per
    row. Without the warm start this takes 2n (measured: 120 for n=60)."""
    import random

    rng = random.Random(0)
    perm = list(range(60))
    rng.shuffle(perm)
    matrix = [[0 if j == perm[i] else rng.randint(1, 9) for j in range(60)] for i in range(60)]
    reads, n = _solver_row_reads(matrix)
    assert reads == n


def test_warm_start_makes_tie_heavy_matrices_cheap():
    """Tie-heavy 200x200: ~600 look-ups with the warm start, ~20,000 without."""
    import random

    rng = random.Random(0)
    matrix = [[rng.randint(0, 3) for _ in range(200)] for _ in range(200)]
    reads, n = _solver_row_reads(matrix)
    assert reads < 5 * n
    assert total(matrix, Munkres().compute(matrix)) == scipy_min_cost(matrix)


def test_performance_large_infeasible_detected_quickly():
    n = 200
    k = n // 2
    # k+1 rows may only use the first k columns: impossible, and large.
    matrix = [[1 if j < k else D for j in range(n)] if i <= k else [1] * n for i in range(n)]
    with pytest.raises(UnsolvableMatrix) as info:
        call_with_timeout(Munkres().compute, 5, matrix)
    assert len(info.value.rows) == k + 1 and len(info.value.cols) == k
