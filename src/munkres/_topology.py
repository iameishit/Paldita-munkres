# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Optimal-solution topology: which edges are forced, optional or excluded.

Using exact dual potentials of one optimal assignment M (re-derived independently
by `_verify`), an allowed cell (i, j) is *tight* iff c_ij = u_i + v_j. Build the
digraph with matched tight edges col->row and unmatched tight edges row->col.
A tight edge lies in some optimal assignment iff its endpoints share a strongly
connected component (equivalently it lies on an alternating cycle); matched edges
whose endpoints are in different components are in *every* optimum (forced).
Rectangular problems are zero-padded to square (dummy rows), which is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._verify import Verification, _exact_rows, _orient, _padded_duals

__all__ = ["Topology", "edge_margins", "topology"]


@dataclass(frozen=True)
class Topology:
    pairs: tuple[tuple[int, int], ...]
    objective: Fraction  # caller's sense
    forced: frozenset[tuple[int, int]]  # in every optimal assignment
    optional: frozenset[tuple[int, int]]  # in some but not every optimal assignment
    excluded_tight: frozenset[tuple[int, int]]  # tight but in no optimal assignment
    exchange_classes: tuple[tuple[tuple[int, int], ...], ...]  # optional edges grouped by SCC
    reduced_cost: dict[tuple[int, int], Fraction]  # c_ij - u_i - v_j >= 0 for every allowed cell
    n_optimal: int | None  # exact number of optimal assignments (None if too large to count)

    @property
    def unique(self) -> bool:
        return not self.optional


def _scc(n_nodes: int, adj: list[list[int]]) -> list[int]:
    """Iterative Tarjan; returns component id per node."""
    index, low, on, comp = [-1] * n_nodes, [0] * n_nodes, [False] * n_nodes, [-1] * n_nodes
    stack: list[int] = []
    counter = ncomp = 0
    for root in range(n_nodes):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, k = work.pop()
            if k == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for t in range(k, len(adj[v])):
                w = adj[v][t]
                if index[w] == -1:
                    work.append((v, t + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return comp


def _count_perfect(rows: list[list[int]], _ncols: int) -> int | None:
    """Number of perfect matchings of a small bipartite graph (bitmask DP); None if too big."""
    k = len(rows)
    if k > 18:
        return None
    dp = {0: 1}
    for r in rows:
        nxt: dict[int, int] = {}
        for mask, ways in dp.items():
            for c in r:
                if not mask >> c & 1:
                    nxt[mask | 1 << c] = nxt.get(mask | 1 << c, 0) + ways
        dp = nxt
    return sum(dp.values())


def topology(matrix: Any, pairs: Any = None, maximize: bool = False) -> Topology:
    """
    Compute the `Topology` of the optimum of `matrix`. If `pairs` is omitted an
    optimal assignment is computed first; supplied pairs must be optimal (checked).
    """
    from munkres._api import solve

    if pairs is None:
        pairs = solve(matrix, maximize=maximize).pairs
    n, m, rows = _exact_rows(matrix, maximize)
    on, om, orows, tr = _orient(n, m, rows)
    mp = [((c, r) if tr else (r, c)) for r, c in pairs]
    res = _padded_duals(on, om, orows, mp)
    if isinstance(res, Verification):
        raise ValueError(f"pairs are not optimal: {res.reason}")
    uu, vv, match, size, full = res
    adj: list[list[int]] = [[] for _ in range(2 * size)]
    red: dict[tuple[int, int], Fraction] = {}
    tight: list[tuple[int, int]] = []
    for i in range(size):
        for j, c in full[i].items():
            r = c - uu[i] - vv[j]
            if i < on:
                red[(i, j)] = r
            if r == 0:
                tight.append((i, j))
                if match[i] == j:
                    adj[size + j].append(i)
                else:
                    adj[i].append(size + j)
    comp = _scc(2 * size, adj)
    forced: set[tuple[int, int]] = set()
    optional: set[tuple[int, int]] = set()
    excl: set[tuple[int, int]] = set()
    classes: dict[int, list[tuple[int, int]]] = {}
    for i, j in tight:
        if i >= on:
            continue
        same = comp[i] == comp[size + j]
        if match[i] == j:
            (optional if same else forced).add((i, j))
        elif same:
            optional.add((i, j))
        else:
            excl.add((i, j))
        if same:
            classes.setdefault(comp[i], []).append((i, j))
    # exact number of optimal assignments of the real rows = product over exchange classes
    count: int | None = 1
    members: dict[int, list[int]] = {}
    for i in range(size):
        members.setdefault(comp[i], []).append(i)
    for cid, rws in members.items():
        if len(rws) == 1 and comp[size + match[rws[0]]] != cid:
            continue
        cols = sorted(
            {
                j
                for i in rws
                for j, c in full[i].items()
                if c - uu[i] - vv[j] == 0 and comp[size + j] == cid
            }
        )
        cmap = {c: k for k, c in enumerate(cols)}
        sub = [
            [cmap[j] for j, c in full[i].items() if c - uu[i] - vv[j] == 0 and j in cmap]
            for i in rws
        ]
        w = _count_perfect(sub, len(cols))
        if w is None:
            count = None
            break
        if count is None:
            raise AssertionError("internal: component count cleared inside the loop")
        count *= w
    if count is not None:
        count //= math.factorial(om - on)  # interchangeable dummy rows
    flip = (lambda p: (p[1], p[0])) if tr else (lambda p: p)
    obj = sum((orows[r][c] for r, c in mp), Fraction(0))
    return Topology(
        tuple(sorted(pairs)),
        -obj if maximize else obj,
        frozenset(map(flip, forced)),
        frozenset(map(flip, optional)),
        frozenset(map(flip, excl)),
        tuple(tuple(sorted(map(flip, v))) for v in classes.values()),
        {flip(k): v for k, v in red.items()},
        count,
    )


def edge_margins(matrix: Any, maximize: bool = False) -> dict[tuple[int, int], dict[str, Any]]:
    """
    Exact per-edge sensitivity by re-solving (O(E) solves; intended for small/medium problems).
    For each allowed cell: `exclude_penalty` = objective increase if the cell is forbidden
    (None if that makes the problem infeasible) and `force_penalty` = increase if the cell
    must be used. Penalties are >= 0 in the caller's direction of "worse".
    """
    from munkres._api import solve
    from munkres._core import DISALLOWED, UnsolvableMatrix, _as_lists, _validated_edges

    rows = _as_lists(matrix)
    base = solve(rows, maximize=maximize)
    n, m, edges = _validated_edges(rows)
    sign = -1 if maximize else 1
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for i, e in enumerate(edges):
        for j, c in e:
            alt = [list(r) for r in rows]
            alt[i][j] = DISALLOWED
            try:
                ex = sign * (Fraction(solve(alt, maximize=maximize).total) - Fraction(base.total))
            except UnsolvableMatrix:
                ex = None
            keep = [[rows[a][b] for b in range(m) if b != j] for a in range(n) if a != i]
            try:
                fr = (
                    sign
                    * (
                        Fraction(c)
                        + (
                            Fraction(solve(keep, maximize=maximize).total)
                            if keep and keep[0]
                            else 0
                        )
                        - Fraction(base.total)
                    )
                    if (
                        (n <= m and keep and keep[0] and len(keep) <= len(keep[0]))
                        or not keep
                        or not keep[0]
                    )
                    else sign
                    * (
                        Fraction(c)
                        + Fraction(solve(keep, maximize=maximize).total)
                        - Fraction(base.total)
                    )
                )
            except UnsolvableMatrix:
                fr = None
            out[(i, j)] = {"exclude_penalty": ex, "force_penalty": fr}
    return out
