# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Connected-component decomposition of the allowed-edge graph (exact, order-independent)."""

from __future__ import annotations

from typing import Any

from munkres._core import UnsolvableMatrix, _as_lists, _validated_edges

__all__ = ["components", "solve_by_components"]


def components(matrix: Any) -> list[tuple[list[int], list[int]]]:
    """Return `[(rows, cols), ...]`, one entry per connected component (isolated vertices included)."""
    n, m, edges = _validated_edges(_as_lists(matrix))
    parent = list(range(n + m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, e in enumerate(edges):
        for j, _ in e:
            parent[find(i)] = find(n + j)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for x in range(n + m):
        g = groups.setdefault(find(x), ([], []))
        (g[0] if x < n else g[1]).append(x if x < n else x - n)
    return sorted(groups.values(), key=lambda g: (g[0][:1] or [10**9], g[1][:1]))


def solve_by_components(
    matrix: Any, maximize: bool = False
) -> tuple[list[tuple[int, int]], float | Any, int]:
    """
    Solve each connected component independently. Returns `(pairs, total, n_components_solved)`.
    Semantics identical to `munkres.solve` (min(n, m) pairs); infeasible components raise
    `UnsolvableMatrix` exactly when the global problem is infeasible.
    """
    from munkres._api import solve

    rows = _as_lists(matrix)
    n = len(rows)
    m = len(rows[0]) if n else 0
    comps = components(rows)
    pairs: list[tuple[int, int]] = []
    total: Any = 0
    solved = 0
    wide = n <= m
    for rs, cs in comps:
        if wide and len(rs) > len(cs):
            raise UnsolvableMatrix(f"rows {rs} can only use {len(cs)} columns (Hall violation)")
        if not wide and len(cs) > len(rs):
            raise UnsolvableMatrix(f"columns {cs} can only use {len(rs)} rows (Hall violation)")
        if not rs or not cs:
            continue
        sub = [[rows[i][j] for j in cs] for i in rs]
        res = solve(sub, maximize=maximize)
        solved += 1
        total = total + res.total
        pairs.extend((rs[a], cs[b]) for a, b in res.pairs)
    return sorted(pairs), total, solved
