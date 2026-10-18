"""munkres.linear_sum_assignment against scipy.optimize.linear_sum_assignment."""

import random

import pytest

from munkres import DISALLOWED, linear_sum_assignment

scipy_optimize = pytest.importorskip("scipy.optimize")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("maximize", [False, True])
def test_totals_match_scipy_on_random_integer_matrices(maximize):
    rng = random.Random(21)
    for _ in range(200):
        r, c = rng.randint(1, 8), rng.randint(1, 8)
        m = [[rng.randint(-20, 20) for _ in range(c)] for _ in range(r)]
        ours = linear_sum_assignment(m, maximize=maximize)
        theirs = scipy_optimize.linear_sum_assignment(m, maximize=maximize)
        arr = np.array(m)
        assert arr[ours[0], ours[1]].sum() == arr[theirs[0], theirs[1]].sum()
        assert len(ours[0]) == len(theirs[0]) == min(r, c)


def test_numpy_in_numpy_out_with_scipy_compatible_dtype():
    arr = np.array([[4.0, 1.0], [2.0, 0.0]])
    ours = linear_sum_assignment(arr)
    theirs = scipy_optimize.linear_sum_assignment(arr)
    assert ours[0].dtype == theirs[0].dtype and ours[0].tolist() == theirs[0].tolist()
    assert ours[1].tolist() == theirs[1].tolist()


def test_both_reject_the_same_invalid_matrices():
    for bad in (
        [[1, float("nan")], [1, 2]],
        [[1, -float("inf")], [1, 2]],
        [[float("inf")] * 2] * 2,
    ):
        with pytest.raises(ValueError, match=r"."):
            scipy_optimize.linear_sum_assignment(bad)
        with pytest.raises(ValueError, match=r"."):
            linear_sum_assignment(bad)


def test_disallowed_is_the_munkres_spelling_of_infinity():
    ours = linear_sum_assignment([[1, DISALLOWED], [DISALLOWED, 1]])
    theirs = scipy_optimize.linear_sum_assignment([[1, np.inf], [np.inf, 1]])
    assert ours == (theirs[0].tolist(), theirs[1].tolist())
