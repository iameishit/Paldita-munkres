# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Tropical (min-plus / max-plus) primitives, an independent subset-DP backend, and closed-form special cases."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _validated_edges

__all__ = [
    "maxplus",
    "minplus",
    "rank_one_decomposition",
    "solve_line_matching",
    "solve_rank_one",
    "tropical_permanent",
]

_INF = float("inf")


def _mp(a: Any, b: Any, pick: Any, neutral: float) -> list[list[Any]]:
    A, B = _as_lists(a), _as_lists(b)
    if A and B and len(A[0]) != len(B):
        raise ValueError("inner dimensions differ")
    return [
        [
            pick((x + y for x, y in zip(row, col, strict=False)), default=neutral)
            for col in zip(*B, strict=False)
        ]
        for row in A
    ]


def minplus(a: Any, b: Any) -> list[list[Any]]:
    """Min-plus matrix product: C[i][j] = min_k (A[i][k] + B[k][j]); +inf entries are neutral."""
    return _mp(a, b, min, _INF)


def maxplus(a: Any, b: Any) -> list[list[Any]]:
    """Max-plus matrix product: C[i][j] = max_k (A[i][k] + B[k][j]); -inf entries are neutral."""
    return _mp(a, b, max, -_INF)


def tropical_permanent(matrix: Any, maximize: bool = False) -> Any | None:
    """
    Min-plus (or max-plus) permanent over injections of all rows into distinct columns, by subset DP
    (O(n 2^m m), m <= 20). An independent computation of the assignment optimum; `None` if infeasible.
    """
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("requires rows <= cols; pass the transpose")
    if m > 20:
        raise ValueError("subset DP supports at most 20 columns")
    pick = max if maximize else min
    dp: dict[int, Any] = {0: 0}
    for i in range(n):
        nxt: dict[int, Any] = {}
        for mask, v in dp.items():
            for j, c in edges[i]:
                if not mask >> j & 1:
                    k = mask | 1 << j
                    nxt[k] = pick(nxt[k], v + c) if k in nxt else v + c
        dp = nxt
    return pick(dp.values()) if dp else None


def rank_one_decomposition(matrix: Any) -> tuple[list[Fraction], list[Fraction]] | None:
    """Exact (a, b) with c[i][j] = a[i] * b[j] for a fully finite matrix, or None if the rank exceeds one."""
    rows = _as_lists(matrix)
    if not rows or any(
        v is DISALLOWED or (isinstance(v, float) and v in (_INF, -_INF)) for r in rows for v in r
    ):
        return None
    c = [[Fraction(v) for v in r] for r in rows]
    n, m = len(c), len(c[0])
    pos = next(((i, j) for i in range(n) for j in range(m) if c[i][j] != 0), None)
    if pos is None:
        return [Fraction(0)] * n, [Fraction(0)] * m
    i0, j0 = pos
    b = [c[i0][j] for j in range(m)]
    a = [c[i][j0] / c[i0][j0] for i in range(n)]
    return (a, b) if all(a[i] * b[j] == c[i][j] for i in range(n) for j in range(m)) else None


def solve_rank_one(matrix: Any) -> tuple[tuple[int, int], ...]:
    """
    Square rank-one matrices (c = a b^T, any signs): the minimum-sum assignment pairs the smallest a with the
    largest b (rearrangement inequality). Raises ValueError if the matrix is not square rank-one.
    """
    rows = _as_lists(matrix)
    dec = rank_one_decomposition(rows)
    if dec is None or len(rows) != len(rows[0]):
        raise ValueError("matrix is not a square rank-one matrix")
    a, b = dec
    ia = sorted(range(len(a)), key=lambda i: (a[i], i))
    jb = sorted(range(len(b)), key=lambda j: (-b[j], j))
    return tuple(sorted(zip(ia, jb, strict=False)))


def solve_line_matching(x: Any, y: Any, p: int = 1) -> tuple[tuple[tuple[int, int], ...], Fraction]:
    """
    Minimum-cost matching of all points x into distinct points y on a line with cost |x - y|**p (p >= 1
    integer, convex => order-preserving optimum). No cost matrix is built when len(x) == len(y).
    Returns `(pairs of original indices, total cost)`.
    """
    xs, ys = [Fraction(v) for v in x], [Fraction(v) for v in y]
    n, m = len(xs), len(ys)
    if n > m:
        raise ValueError("requires len(x) <= len(y)")
    if p < 1 or int(p) != p:
        raise ValueError("p must be an integer >= 1")
    ox, oy = sorted(range(n), key=lambda i: (xs[i], i)), sorted(range(m), key=lambda j: (ys[j], j))
    if n == m:
        pairs = list(zip(ox, oy, strict=False))
    else:

        def cost(i: int, j: int) -> Any:
            return abs(xs[ox[i]] - ys[oy[j]]) ** p

        dp: list[list[Fraction | None]] = [[None] * (m + 1) for _ in range(n + 1)]
        take = [[False] * (m + 1) for _ in range(n + 1)]
        for j in range(m + 1):
            dp[0][j] = Fraction(0)
        for i in range(1, n + 1):
            for j in range(i, m + 1):
                best, took = dp[i][j - 1], False
                prev = dp[i - 1][j - 1]
                if prev is not None:
                    cand = prev + cost(i - 1, j - 1)
                    if best is None or cand < best:
                        best, took = cand, True
                dp[i][j], take[i][j] = best, took
        pairs, j = [], m
        for i in range(n, 0, -1):
            while not take[i][j]:
                j -= 1
            pairs.append((ox[i - 1], oy[j - 1]))
            j -= 1
    total = sum((abs(xs[i] - ys[j]) ** p for i, j in pairs), Fraction(0))
    return tuple(sorted(pairs)), total
