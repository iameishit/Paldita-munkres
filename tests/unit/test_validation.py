"""Input validation and the DISALLOWED sentinel, in isolation."""

import copy
import pickle

import pytest

from munkres import DISALLOWED, Munkres
from munkres._core import DISALLOWED_OBJ, _as_lists, _validated_edges

INF = float("inf")


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        ([[1, 2], [3]], ValueError, "not rectangular: row 1 has 1 entries but row 0 has 2"),
        ([[1, float("nan")]], ValueError, r"cell \[0\]\[1\] is NaN"),
        ([[1, -INF]], ValueError, "-infinity"),
        ([[1, "x"]], TypeError, "not a number"),
        ([[1, None]], TypeError, "not a number"),
        ([[1, 1j]], TypeError, "not a number"),
        ("abc", TypeError, "sequence of rows"),
        (5, TypeError, "sequence of rows"),
        ([1, 2, 3], TypeError, "sequence of rows"),
    ],
)
def test_invalid_input_is_rejected_with_a_specific_message(matrix, error, message):
    with pytest.raises(error, match=message):
        Munkres().compute(matrix)


def test_validated_edges_lists_only_allowed_cells():
    n, m, edges = _validated_edges([[1, DISALLOWED, 3], [INF, 5, 6]])
    assert (n, m) == (2, 3)
    assert edges == [[(0, 1), (2, 3)], [(1, 5), (2, 6)]]


def test_as_lists_returns_an_independent_copy():
    original = [[1, 2], [3, 4]]
    copied = _as_lists(original)
    copied[0][0] = 99
    assert original == [[1, 2], [3, 4]]
    assert _as_lists(((1, 2), (3, 4))) == [[1, 2], [3, 4]]


def test_sentinel_is_a_true_singleton():
    assert DISALLOWED_OBJ() is DISALLOWED
    assert repr(DISALLOWED) == "DISALLOWED"
    assert copy.copy(DISALLOWED) is DISALLOWED
    assert copy.deepcopy([DISALLOWED])[0] is DISALLOWED
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(DISALLOWED, protocol)) is DISALLOWED


def test_empty_inputs_return_an_empty_result():
    assert Munkres().compute([]) == []
    assert Munkres().compute([[]]) == []
    assert Munkres().compute([[], []]) == []
