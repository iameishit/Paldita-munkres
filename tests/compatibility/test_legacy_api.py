"""The 1.x API keeps working."""

import munkres
from munkres import DISALLOWED, Munkres, UnsolvableMatrix, make_cost_matrix

from ..reference.munkres_reference import min_cost_assignment


def test_star_import_exposes_the_documented_names():
    namespace = {}
    exec("from munkres import *", namespace)  # noqa: S102
    for name in ("Munkres", "make_cost_matrix", "DISALLOWED", "UnsolvableMatrix", "print_matrix"):
        assert name in namespace


def test_module_metadata():
    assert munkres.__license__ == "Apache-2.0"
    assert munkres.__url__.startswith("https://")
    assert munkres.__version__.startswith("2.")
    assert "Clapper" in munkres.__author__


def test_classic_workflow_still_works():
    matrix = [[5, 9, 1], [10, 3, 2], [8, 7, 4]]
    pairs = Munkres().compute(matrix)
    assert sum(matrix[r][c] for r, c in pairs) == min_cost_assignment(matrix)[0]
    profit = make_cost_matrix(matrix)
    best = Munkres().compute(profit)
    assert sum(matrix[r][c] for r, c in best) == 23


def test_unsolvable_matrix_can_be_caught_as_before():
    try:
        Munkres().compute([[DISALLOWED, DISALLOWED], [1, 2]])
    except UnsolvableMatrix as caught:
        assert isinstance(caught, Exception) and isinstance(caught, ValueError)
    else:
        raise AssertionError("expected UnsolvableMatrix")


def test_one_instance_can_be_reused_for_many_problems():
    solver = Munkres()
    for _ in range(3):
        assert solver.compute([[4, 1], [2, 0]]) == [(0, 1), (1, 0)]
        assert solver.compute([[1]]) == [(0, 0)]
