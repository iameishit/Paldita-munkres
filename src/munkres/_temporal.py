# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Assignment with inertia (Bell Labs BSTJ 1982-style rearrangement cost).

Model: minimise  sum_i c[i][a(i)] + sum_i lambda_i * [a(i) != previous(i)]  over assignments a,
where the switching penalty is *separable per row*. That is exactly an ordinary assignment on
the cost matrix c'[i][j] = c[i][j] + lambda_i * [j != previous(i)], so it is solved to proven
optimality. `TemporalAssignment.step` applies this greedily stage by stage (a rolling-horizon
policy): it is optimal for each stage given the past, NOT a global optimum over all stages.
Rows with no previous column, and unmatched rows (tall matrices), incur no penalty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from munkres._core import _as_lists, _is_forbidden

__all__ = ["InertiaResult", "TemporalAssignment", "assign_with_inertia"]


@dataclass(frozen=True)
class InertiaResult:
    pairs: tuple[tuple[int, int], ...]
    cost: Any  # sum of c over pairs (no penalty)
    penalty: Any  # total switching penalty paid
    objective: Any  # cost + penalty (what was minimised; profit - penalty if maximize)
    switches: int


def assign_with_inertia(
    cost: Any,
    previous: Mapping[int, int | None] | Sequence[int | None] | None,
    switch_cost: Any = 0,
    maximize: bool = False,
) -> InertiaResult:
    """Optimal assignment that also pays `switch_cost` (scalar or per-row) for each row moved off its `previous` column."""
    from munkres._api import solve

    rows = _as_lists(cost)
    n = len(rows)
    prev: dict[int, int | None]
    if previous is None:
        prev = {}
    elif isinstance(previous, Mapping):
        prev = dict(previous)
    else:
        prev = dict(enumerate(previous))
    lam = [switch_cost] * n if not isinstance(switch_cost, Sequence) else list(switch_cost)
    if len(lam) != n:
        raise ValueError("switch_cost must be a scalar or one value per row")
    if any(x < 0 for x in lam):
        raise ValueError("switch costs must be non-negative")
    sign = -1 if maximize else 1
    adj = [
        [
            x
            if _is_forbidden(x)
            else x + sign * lam[i] * (1 if prev.get(i) is not None and prev[i] != j else 0)
            for j, x in enumerate(row)
        ]
        for i, row in enumerate(rows)
    ]
    res = solve(adj, maximize=maximize)
    pairs = tuple(sorted(res.pairs))
    base = sum((rows[i][j] for i, j in pairs), 0)
    sw = [(i, j) for i, j in pairs if prev.get(i) is not None and prev[i] != j]
    pen = sum((lam[i] for i, _ in sw), 0)
    return InertiaResult(pairs, base, pen, base - pen if maximize else base + pen, len(sw))


class TemporalAssignment:
    """Rolling-horizon assignment with switching penalty; keeps the previous stage's pairs."""

    def __init__(self, switch_cost: Any = 0, maximize: bool = False) -> None:
        self.switch_cost = switch_cost
        self.maximize = maximize
        self.previous: dict[int, int] = {}
        self.history: list[InertiaResult] = []

    def step(self, cost: Any) -> InertiaResult:
        res = assign_with_inertia(cost, self.previous, self.switch_cost, self.maximize)
        self.previous = dict(res.pairs)
        self.history.append(res)
        return res
