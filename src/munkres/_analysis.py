# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Analysis of an assignment problem: prices, what-ifs, runners-up, bottleneck."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any

from munkres._api import Assignment, _make_assignment, _negated, solve
from munkres._core import (
    DISALLOWED,
    MatrixLike,
    UnsolvableMatrix,
    _as_lists,
    _assign,
    _validated_edges,
)

__all__ = ["Prices", "bottleneck", "counterfactual", "k_best", "shadow_prices", "tolerance"]


@dataclass(frozen=True)
class Prices:
    """
    LP dual variables of the assignment problem ("shadow prices").

    For every allowed cell, `cost[i][j] >= row_prices[i] + col_prices[j]`, with
    equality on every pair of the optimal assignment, and
    `total == sum(row_prices) + sum(col_prices) ==` the optimal cost. That
    equality is a *proof of optimality* anyone can re-check in O(rows * columns).
    A row price says how much one more unit of "demand" at that row would cost.
    """

    row_prices: tuple[Any, ...]
    col_prices: tuple[Any, ...]
    total: Any


def shadow_prices(matrix: MatrixLike) -> Prices:
    """Solve the problem and return its dual variables (see `Prices`)."""
    n_rows, n_cols, edges = _validated_edges(matrix)
    duals: list[Any] = []
    _assign(n_rows, n_cols, edges, duals=duals)
    u, v = duals
    rows_p, cols_p = (u, v) if n_rows <= n_cols else (v, u)
    return Prices(tuple(rows_p), tuple(cols_p), sum(rows_p) + sum(cols_p))


def _check_cell(matrix: list[list[Any]], row: int, col: int) -> None:
    n_rows, n_cols, edges = _validated_edges(matrix)
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise ValueError(f"cell ({row}, {col}) is outside the {n_rows}x{n_cols} matrix")
    if all(c != col for c, _ in edges[row]):
        raise ValueError(f"cell ({row}, {col}) is forbidden")


def counterfactual(matrix: MatrixLike, row: int, col: int) -> Any:
    """
    "What would it cost to force `row` onto `col`?" Returns how much the optimal
    total rises if that pair is made mandatory (0 if some optimal assignment
    already contains it). Raises `UnsolvableMatrix` if forcing it makes the
    problem impossible, and `ValueError` for a forbidden or out-of-range cell.
    """
    rows = _as_lists(matrix)
    _check_cell(rows, row, col)
    best = solve(rows).total
    forced = [list(r) for r in rows]
    for j in range(len(rows[0])):
        if j != col:
            forced[row][j] = DISALLOWED
    for i in range(len(rows)):
        if i != row:
            forced[i][col] = DISALLOWED
    return solve(forced).total - best


def tolerance(matrix: MatrixLike, row: int, col: int) -> Any:
    """
    How much can the cost of a pair in the optimal assignment *rise* before the
    assignment has to change? (the optimum without that pair, minus the optimum).
    `float('inf')` if the pair can never be replaced. `ValueError` if the pair is
    not in the assignment `solve` returns.
    """
    rows = _as_lists(matrix)
    _check_cell(rows, row, col)
    chosen = solve(rows)
    if (row, col) not in chosen.pairs:
        raise ValueError(f"({row}, {col}) is not in the optimal assignment")
    without = [list(r) for r in rows]
    without[row][col] = DISALLOWED
    try:
        return solve(without).total - chosen.total
    except UnsolvableMatrix:
        return float("inf")


def k_best(matrix: MatrixLike, k: int, *, maximize: bool = False) -> list[Assignment]:
    """
    The `k` best complete assignments, best first (Murty's algorithm). Fewer are
    returned if fewer exist. Each is a distinct set of pairs; ties are ordered
    arbitrarily.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    original = _as_lists(matrix)
    work = _negated(original) if maximize else original
    n_rows, n_cols, _ = _validated_edges(work)
    found: list[Assignment] = []
    if k == 0:
        return found

    def constrained(forbid: tuple[tuple[int, int], ...], force: tuple[tuple[int, int], ...]) -> Any:
        grid = [list(r) for r in work]
        for i, j in forbid:
            grid[i][j] = DISALLOWED
        for i, j in force:
            for jj in range(n_cols):
                if jj != j:
                    grid[i][jj] = DISALLOWED
            for ii in range(n_rows):
                if ii != i:
                    grid[ii][j] = DISALLOWED
        try:
            return solve(grid)
        except UnsolvableMatrix:
            return None

    counter = itertools.count()
    heap: list[Any] = []
    first = constrained((), ())
    if first is not None:
        heapq.heappush(heap, (first.total, next(counter), first, (), ()))
    while heap and len(found) < k:
        _, _, node, forbid, force = heapq.heappop(heap)
        pairs = list(node.pairs)
        found.append(_make_assignment(original, pairs, (n_rows, n_cols), maximize=maximize))
        for t, pair in enumerate(pairs):
            child_force = force + tuple(pairs[:t])
            child_forbid = (*forbid, pair)
            child = constrained(child_forbid, child_force)
            if child is not None:
                heapq.heappush(heap, (child.total, next(counter), child, child_forbid, child_force))
    return found


def bottleneck(matrix: MatrixLike, *, maximize: bool = False) -> Assignment:
    """
    The assignment whose *single worst pair* is as good as possible (minimise the
    largest cost; with `maximize=True`, maximise the smallest profit). Among
    assignments that tie on the worst pair, the lowest total (highest, when
    maximising) wins. Good for fairness: "nobody gets a terrible job".
    Raises `UnsolvableMatrix` if no complete assignment exists.
    """
    original = _as_lists(matrix)
    work = _negated(original) if maximize else original
    n_rows, n_cols, edges = _validated_edges(work)
    shape = (n_rows, n_cols)
    values = sorted({c for allowed in edges for _, c in allowed})
    if not values:
        _assign(n_rows, n_cols, edges)  # raises if rows exist but nothing is allowed
        return _make_assignment(original, [], shape, maximize=maximize)

    def capped(limit: Any) -> list[list[tuple[int, Any]]]:
        return [[(j, c) for j, c in allowed if c <= limit] for allowed in edges]

    def feasible(limit: Any) -> bool:
        try:
            _assign(n_rows, n_cols, capped(limit))
        except UnsolvableMatrix:
            return False
        return True

    if not feasible(values[-1]):
        _assign(n_rows, n_cols, edges)  # raise the real, explanatory error
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(values[mid]):
            hi = mid
        else:
            lo = mid + 1
    pairs = _assign(n_rows, n_cols, capped(values[lo]))
    return _make_assignment(original, pairs, shape, maximize=maximize)
