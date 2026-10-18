"""make_cost_matrix, pad_matrix and print_matrix (the 1.x helpers)."""

from decimal import Decimal

from munkres import DISALLOWED, Munkres, make_cost_matrix, print_matrix


def test_make_cost_matrix_default_and_custom_inversion():
    assert make_cost_matrix([[1, 2], [3, 4]]) == [[3, 2], [1, 0]]
    assert make_cost_matrix([[1, 2], [3, 4]], lambda x: -x) == [[-1, -2], [-3, -4]]
    assert make_cost_matrix([]) == []


def test_make_cost_matrix_carries_forbidden_cells_through():
    out = make_cost_matrix([[5, DISALLOWED], [2, float("inf")]])
    assert out[0][0] == 0 and out[1][0] == 3
    assert out[0][1] is DISALLOWED and out[1][1] == float("inf")
    assert make_cost_matrix([[DISALLOWED]]) == [[DISALLOWED]]


def test_make_cost_matrix_is_exact_for_decimal():
    assert make_cost_matrix([[Decimal("0.1"), Decimal("0.3")]]) == [
        [Decimal("0.2"), Decimal("0.0")]
    ]


def test_pad_matrix_makes_a_square_copy():
    original = [[1, 2], [3, 4], [5, 6]]
    padded = Munkres().pad_matrix(original)
    assert padded == [[1, 2, 0], [3, 4, 0], [5, 6, 0]]
    assert original == [[1, 2], [3, 4], [5, 6]]
    assert Munkres().pad_matrix([[1, 2, 3]], pad_value=9) == [[1, 2, 3], [9, 9, 9], [9, 9, 9]]


def test_print_matrix_shows_forbidden_cells_as_D(capsys):
    print_matrix([[1, DISALLOWED], [30, 4]])
    out = capsys.readouterr().out
    assert "D" in out and "30" in out and out.count("[") == 2
