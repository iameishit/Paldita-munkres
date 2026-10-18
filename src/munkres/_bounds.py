# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""LAP as a building block inside other optimisers: bound-aware solving, reduced-cost fixing, QAP bounds."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _validated_edges

__all__ = [
    "BoundResult",
    "prune_edges_by_bound",
    "qap_cost",
    "qap_local_search",
    "qap_lower_bound",
    "solve_with_bound",
]


@dataclass(frozen=True)
class BoundResult:
    """Result of `solve_with_bound`."""

    pairs: tuple[tuple[int, int], ...] | None  # None: no assignment with objective <= upper exists
    total: Any
    solved: bool  # False if the cheap lower bound already proved infeasibility (no LAP was solved)


def solve_with_bound(cost: Any, upper: Any, maximize: bool = False) -> BoundResult:
    """
    Minimisation: an optimal assignment if its cost is <= `upper`, else `pairs=None`. When rows <= cols the
    sum of row minima is checked first and can reject without solving. (maximize: objective >= `upper`.)
    """
    from munkres._api import solve

    rows = _as_lists(cost)
    n, m, edges = _validated_edges(rows)
    if not maximize and n <= m and n:
        lb = sum((min(c for _, c in e) for e in edges if e), 0)
        if lb > upper:
            return BoundResult(None, None, False)
    res = solve(rows, maximize=maximize)
    ok = res.total >= upper if maximize else res.total <= upper
    return BoundResult(tuple(sorted(res.pairs)) if ok else None, res.total if ok else None, True)


def prune_edges_by_bound(cost: Any, upper: Any) -> tuple[list[list[Any]], int]:
    """
    Reduced-cost fixing for minimisation: forbid every cell (i, j) with  optimum + reduced_cost(i, j) > upper.
    Any assignment using such a cell costs more than `upper`, so all assignments with cost <= upper survive.
    Returns `(pruned matrix, number of cells removed)`.
    """
    from munkres._topology import topology

    rows = [list(r) for r in _as_lists(cost)]
    t = topology(rows)
    removed = 0
    for (i, j), r in t.reduced_cost.items():
        if t.objective + r > upper:
            rows[i][j] = DISALLOWED
            removed += 1
    return rows, removed


def qap_cost(flow: Any, dist: Any, perm: Any) -> Any:
    """Koopmans-Beckmann QAP objective: sum_{i,j} flow[i][j] * dist[perm[i]][perm[j]]."""
    n = len(perm)
    return sum(flow[i][j] * dist[perm[i]][perm[j]] for i in range(n) for j in range(n))


def qap_lower_bound(flow: Any, dist: Any) -> Fraction:
    """
    Gilmore-Lawler lower bound: L[i][k] = f_ii d_kk + (smallest inner product of row i of F and row k of D off the
    diagonal, by the rearrangement inequality); the bound is the optimum of the LAP on L. A valid bound for
    min over permutations of `qap_cost`.
    """
    from munkres._api import solve

    F, D = _as_lists(flow), _as_lists(dist)
    n = len(F)
    L = []
    for i in range(n):
        fi = sorted(Fraction(F[i][j]) for j in range(n) if j != i)
        row = []
        for k in range(n):
            dk = sorted((Fraction(D[k][c2]) for c2 in range(n) if c2 != k), reverse=True)
            row.append(
                Fraction(F[i][i]) * Fraction(D[k][k])
                + sum((a * b for a, b in zip(fi, dk, strict=False)), Fraction(0))
            )
        L.append(row)
    return Fraction(solve(L).total)


def qap_local_search(flow: Any, dist: Any, perm: Any) -> tuple[tuple[int, ...], Any]:
    """First-improvement 2-swap local search from `perm`. A heuristic: the result is a local optimum, not proven optimal."""
    p = list(perm)
    best = qap_cost(flow, dist, p)
    improved = True
    while improved:
        improved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                p[i], p[j] = p[j], p[i]
                c = qap_cost(flow, dist, p)
                if c < best:
                    best, improved = c, True
                else:
                    p[i], p[j] = p[j], p[i]
    return tuple(p), best
