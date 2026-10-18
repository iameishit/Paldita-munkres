# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Interval-uncertain assignment: regret of an assignment and its worst-case scenario.

Costs are intervals [low, high]. For an assignment x the regret is
    max over scenarios s in the box of  cost(x, s) - OPT(s).
For 0-1 problems such as assignment the maximum is attained at the scenario that sets the
cells *in x* to `high` and all other cells to `low` (Averbakh 2001), so `regret_of` needs one LAP.
`minimax_regret_exact` is strongly NP-hard in general; this version enumerates all assignments
and is limited to small n (default 7).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from munkres._core import DISALLOWED

__all__ = ["Regret", "minimax_regret_exact", "regret_of"]


@dataclass(frozen=True)
class Regret:
    value: Any  # worst-case regret of the assignment
    scenario: list[list[Any]]  # the worst-case cost matrix (certificate)
    optimal_pairs: tuple[tuple[int, int], ...]  # an optimum under that scenario


def _validate_regret_inputs(
    pairs: Any, low: Any, high: Any
) -> tuple[list[tuple[int, int]], int, int]:
    """Reject anything that is not a complete, injective assignment inside well-formed interval matrices."""
    import operator

    n = len(low)
    m = len(low[0]) if n else 0
    if len(high) != n or any(len(r) != m for r in low) or any(len(r) != m for r in high):
        raise ValueError("low and high must be rectangular matrices of the same shape")
    if n > m:
        raise ValueError("regret needs rows <= cols; pass the transpose")
    try:
        plist = [(operator.index(i), operator.index(j)) for i, j in pairs]
    except (TypeError, ValueError):
        raise ValueError("pairs must be (row, col) integer pairs") from None
    if any(not (0 <= i < n and 0 <= j < m) for i, j in plist):
        raise ValueError("pair index out of range")
    rows_used = [i for i, _ in plist]
    cols_used = [j for _, j in plist]
    if len(set(rows_used)) != len(rows_used) or len(set(cols_used)) != len(cols_used):
        raise ValueError("pairs must not repeat a row or a column")
    if len(plist) != n:
        raise ValueError(
            f"pairs must assign every one of the {n} rows exactly once, got {len(plist)}"
        )
    for i in range(n):
        for j in range(m):
            lo, hi = low[i][j], high[i][j]
            if lo is DISALLOWED or hi is DISALLOWED:
                continue
            if lo > hi:
                raise ValueError(f"interval at {(i, j)} has low > high")
    return plist, n, m


def regret_of(pairs: Any, low: Any, high: Any) -> Regret:
    """
    Exact worst-case regret of `pairs` and the scenario realising it (a checkable certificate).

    `pairs` must be a complete assignment (every row once, no column twice) and every interval must
    satisfy low <= high; otherwise `ValueError`. Under those conditions the regret is never negative.
    """
    from munkres._api import solve

    plist, n, m = _validate_regret_inputs(pairs, low, high)
    chosen = set(plist)
    scen = [
        [
            DISALLOWED
            if low[i][j] is DISALLOWED or high[i][j] is DISALLOWED
            else (high[i][j] if (i, j) in chosen else low[i][j])
            for j in range(m)
        ]
        for i in range(n)
    ]
    for i, j in plist:
        if scen[i][j] is DISALLOWED:
            raise ValueError(f"pair {(i, j)} uses a forbidden cell")
    opt = solve(scen)
    mine = sum(scen[i][j] for i, j in plist)
    return Regret(mine - opt.total, scen, tuple(sorted(opt.pairs)))


def minimax_regret_exact(
    low: Any, high: Any, max_n: int = 7
) -> tuple[tuple[tuple[int, int], ...], Any]:
    """Assignment (rows <= cols) minimising worst-case regret, by enumeration. Raises if n > max_n."""
    n, m = len(low), len(low[0]) if low else 0
    if n > max_n or n > m:
        raise ValueError(f"exact minimax regret supports rows <= cols and n <= {max_n}")
    best = None
    for cols in itertools.permutations(range(m), n):
        pairs = tuple(enumerate(cols))
        if any(low[i][j] is DISALLOWED or high[i][j] is DISALLOWED for i, j in pairs):
            continue
        r = regret_of(pairs, low, high).value
        if best is None or r < best[1]:
            best = (pairs, r)
    if best is None:
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible assignment")
    return best
