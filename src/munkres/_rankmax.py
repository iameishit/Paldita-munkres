# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Rank-maximal matching with minimum-cost tie-break (Irving et al. 2006 objective).

Rows rank columns (rank 1 = best, None = unacceptable). A matching's *signature* is
(s_1, ..., s_R), s_r = number of rows matched at rank r. The result maximises the
signature lexicographically, then minimises the optional cost. Rows may stay unmatched.
Reduction to an ordinary assignment: weight w_r = N^(R-r) with N = min(n, m) + 1 makes
lexicographic comparison equal to numeric comparison; the cost tie-break is scaled below
one signature unit; unmatched rows take zero-profit dummy columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED

__all__ = ["RankMaximal", "rank_maximal"]


@dataclass(frozen=True)
class RankMaximal:
    pairs: tuple[tuple[int, int], ...]
    signature: tuple[int, ...]
    cost: Any


def rank_maximal(ranks: Any, cost: Any = None) -> RankMaximal:
    """Rank-maximal matching: lexicographically most first choices, then second choices, ..., then least `cost`."""
    from munkres._api import solve

    rk = [list(r) for r in ranks]
    n = len(rk)
    m = len(rk[0]) if n else 0
    if any(len(r) != m for r in rk):
        raise ValueError("rank matrix must be rectangular")
    for r in rk:
        for x in r:
            if x is not None and (not isinstance(x, int) or x < 1):
                raise ValueError("ranks must be positive ints or None")
    top = max((x for r in rk for x in r if x is not None), default=0)
    if n == 0 or m == 0 or top == 0:
        return RankMaximal((), tuple([0] * top), 0)
    c = [[0] * m for _ in range(n)] if cost is None else [list(r) for r in cost]
    big_n = min(n, m) + 1
    span = 1 + sum(
        abs(Fraction(c[i][j])) for i in range(n) for j in range(m) if rk[i][j] is not None
    )
    profit: list[list[Any]] = []
    for i in range(n):
        row: list[Any] = []
        for j in range(m):
            r = rk[i][j]
            row.append(
                DISALLOWED if r is None else big_n ** (top - r) * span * 2 - Fraction(c[i][j])
            )
        row.extend([Fraction(0)] * n)  # dummy columns: "stay unmatched" at zero profit
        profit.append(row)
    res = solve(profit, maximize=True)
    pairs = tuple(sorted((i, j) for i, j in res.pairs if j < m))
    sig = [0] * top
    for i, j in pairs:
        sig[rk[i][j] - 1] += 1
    return RankMaximal(pairs, tuple(sig), sum((c[i][j] for i, j in pairs), 0))
