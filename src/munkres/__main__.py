# Copyright (c) 2008-2020 Brian M. Clapper; (c) 2026 Eishit Nigam
# Licensed under the Apache License, Version 2.0. See LICENSE.md and NOTICE.
"""``python -m munkres`` -- with no arguments, solve a set of built-in example
matrices and check each answer against its known optimum (an installation smoke
test); with arguments, the command line interface (see ``python -m munkres -h``)."""

from __future__ import annotations

from typing import Any

from munkres import DISALLOWED, Munkres, print_matrix

D = DISALLOWED

EXAMPLES: list[tuple[list[list[Any]], float]] = [
    # Square
    ([[400, 150, 400], [400, 450, 600], [300, 225, 300]], 850),
    # Rectangular variant
    ([[400, 150, 400, 1], [400, 450, 600, 2], [300, 225, 300, 3]], 452),
    # Square
    ([[10, 10, 8], [9, 8, 1], [9, 7, 4]], 18),
    # Square variant with floating point value
    ([[10.1, 10.2, 8.3], [9.4, 8.5, 1.6], [9.7, 7.8, 4.9]], 19.5),
    # Rectangular variant
    ([[10, 10, 8, 11], [9, 8, 1, 1], [9, 7, 4, 10]], 15),
    # Rectangular variant with floating point value
    ([[10.01, 10.02, 8.03, 11.04], [9.05, 8.06, 1.07, 1.08], [9.09, 7.1, 4.11, 10.12]], 15.2),
    # Rectangular with DISALLOWED
    ([[4, 5, 6, D], [1, 9, 12, 11], [D, 5, 4, D], [12, 12, 12, 10]], 20),
    # Rectangular variant with DISALLOWED and floating point value
    (
        [
            [4.001, 5.002, 6.003, D],
            [1.004, 9.005, 12.006, 11.007],
            [D, 5.008, 4.009, D],
            [12.01, 12.011, 12.012, 10.013],
        ],
        20.028,
    ),
    # DISALLOWED to force pairings
    ([[1, D, D, D], [D, 2, D, D], [D, D, 3, D], [D, D, D, 4]], 10),
    # DISALLOWED to force pairings with floating point value
    ([[1.1, D, D, D], [D, 2.2, D, D], [D, D, 3.3, D], [D, D, D, 4.4]], 11.0),
]


def main() -> None:
    solver = Munkres()
    for cost_matrix, expected_total in EXAMPLES:
        print_matrix(cost_matrix, msg="cost matrix")
        total_cost: Any = 0
        for r, c in solver.compute(cost_matrix):
            value = cost_matrix[r][c]
            total_cost += value
            print(f"({r}, {c}) -> {value}")
        print(f"lowest cost={total_cost}")
        if expected_total != total_cost:
            raise SystemExit(f"self-check failed: expected {expected_total}, got {total_cost}")


if __name__ == "__main__":
    from munkres.cli import main as cli_main

    raise SystemExit(cli_main())
