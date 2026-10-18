# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
The optimal-solution landscape: all optima, canonical optimum, face dimension, cost reduction,
Dulmage-Mendelsohn missable sets. Everything is exact (rational arithmetic) and meant for
small/medium problems; enumeration helpers take a `limit` and raise if it is exceeded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _is_forbidden, _validated_edges

__all__ = [
    "DulmageMendelsohn",
    "ReducedProblem",
    "canonical_optimum",
    "dulmage_mendelsohn",
    "enumerate_optima",
    "exchange_cycles",
    "face_dimension",
    "matching_distance",
    "most_different_optimum",
    "reduce_costs",
]


def enumerate_optima(
    matrix: Any, maximize: bool = False, limit: int = 100_000
) -> Iterator[tuple[tuple[int, int], ...]]:
    """
    Yield every optimal assignment (as sorted pair tuples), exactly once.
    Uses the fact that an assignment is optimal iff it only uses *tight* cells (reduced cost 0)
    and covers every column with a non-zero dual. Raises `RuntimeError` after `limit` solutions.
    """
    from munkres._api import solve
    from munkres._verify import Certificate, _exact_rows, _orient, certify

    pairs = solve(matrix, maximize=maximize).pairs
    cert = certify(matrix, pairs, maximize)
    if not isinstance(cert, Certificate):
        raise AssertionError(cert.reason)
    n, m, rows = _exact_rows(matrix, maximize)
    on, om, orows, tr = _orient(n, m, rows)
    tight = [[j for j, c in orows[i].items() if c - cert.u[i] - cert.v[j] == 0] for i in range(on)]
    must = [j for j in range(om) if cert.v[j] != 0]
    emitted = 0

    def rec(i: int, used: int, acc: list[tuple[int, int]]) -> Iterator[tuple[tuple[int, int], ...]]:
        nonlocal emitted
        if i == on:
            if all(used >> j & 1 for j in must):
                emitted += 1
                if emitted > limit:
                    raise RuntimeError(f"more than {limit} optimal assignments")
                yield tuple(sorted((c, r) if tr else (r, c) for r, c in acc))
            return
        if sum(1 for j in must if not used >> j & 1) > on - i:
            return
        for j in tight[i]:
            if not used >> j & 1:
                acc.append((i, j))
                yield from rec(i + 1, used | 1 << j, acc)
                acc.pop()

    yield from rec(0, 0, [])


def face_dimension(matrix: Any, maximize: bool = False, limit: int = 20_000) -> int:
    """Dimension of the face of the assignment polytope spanned by all optimal assignments (0 = unique optimum)."""
    sols = list(enumerate_optima(matrix, maximize, limit))
    cells = sorted({p for s in sols for p in s})
    idx = {p: k for k, p in enumerate(cells)}
    base = {idx[p] for p in sols[0]}
    vecs = []
    for s in sols[1:]:
        cur = {idx[p] for p in s}
        v = [Fraction(0)] * len(cells)
        for k in cur - base:
            v[k] += 1
        for k in base - cur:
            v[k] -= 1
        vecs.append(v)
    rank = 0
    for col in range(len(cells)):
        piv = next((r for r in range(rank, len(vecs)) if vecs[r][col] != 0), None)
        if piv is None:
            continue
        vecs[rank], vecs[piv] = vecs[piv], vecs[rank]
        for r in range(len(vecs)):
            if r != rank and vecs[r][col] != 0:
                f = vecs[r][col] / vecs[rank][col]
                vecs[r] = [a - f * b for a, b in zip(vecs[r], vecs[rank], strict=False)]
        rank += 1
    return rank


def matching_distance(a: Any, b: Any) -> int:
    """Number of pairs of assignment `a` that are not in `b` (rows that must be re-assigned to reach `b`)."""
    return len(set(map(tuple, a)) - set(map(tuple, b)))


def most_different_optimum(
    matrix: Any, reference: Any, maximize: bool = False, limit: int = 100_000
) -> tuple[tuple[int, int], ...]:
    """Optimal assignment farthest (by `matching_distance`) from `reference`; ties -> lexicographically smallest."""
    return min(
        enumerate_optima(matrix, maximize, limit),
        key=lambda s: (-matching_distance(s, reference), s),
    )


def canonical_optimum(matrix: Any, maximize: bool = False) -> tuple[tuple[int, int], ...]:
    """
    The lexicographically smallest optimal assignment (compare the column chosen by row 0, then row 1, ...).
    Exact: solved on costs perturbed by infinitesimally small rational weights, so it is deterministic and
    independent of solver tie-breaking. Requires rows <= cols.
    """
    from munkres._api import solve

    rows = _as_lists(matrix)
    n = len(rows)
    m = len(rows[0]) if n else 0
    if n > m:
        raise ValueError("canonical_optimum requires rows <= cols; pass the transpose")
    if n == 0:
        return ()
    base = [[None if (_is_forbidden(v)) else Fraction(v) for v in r] for r in rows]
    if maximize:
        base = [[None if v is None else -v for v in r] for r in base]
    den = 1
    for r in base:
        for v in r:
            if v is not None:
                den = lcm(den, v.denominator)
    scale = (m + 1) ** n
    delta = Fraction(1, 2 * den * scale * (m + 1))  # smaller than any objective gap (>= 1/den)
    pert = [
        [
            DISALLOWED if v is None else v + delta * j * (m + 1) ** (n - 1 - i)
            for j, v in enumerate(r)
        ]
        for i, r in enumerate(base)
    ]
    return tuple(sorted(solve(pert).pairs))


@dataclass(frozen=True)
class ReducedProblem:
    """`matrix` is non-negative; `original_objective = reduced_objective + constant`."""

    matrix: list[list[Any]]
    row_offsets: tuple[Any, ...]  # subtracted from each row (all zeros if not applicable)
    col_offsets: tuple[Any, ...]
    constant: Any


def reduce_costs(matrix: Any) -> ReducedProblem:
    """
    Subtract row minima (valid when rows <= cols: every row is used) and then column minima
    (valid when rows >= cols: every column is used). The optimal *assignment* is unchanged and the
    objective differs by exactly `constant`. Forbidden cells are preserved. Exact for ints/Fractions.
    """
    rows = _as_lists(matrix)
    n, m, _edges = _validated_edges(rows)
    out: list[list[Any]] = [list(r) for r in rows]
    ro: list[Any] = [0] * n
    co: list[Any] = [0] * m

    def allowed(v: Any) -> bool:
        return not (_is_forbidden(v))

    if n <= m:
        for i in range(n):
            vals = [v for v in out[i] if allowed(v)]
            if vals:
                ro[i] = min(vals)
                out[i] = [v - ro[i] if allowed(v) else v for v in out[i]]
    if n >= m:
        for j in range(m):
            vals = [out[i][j] for i in range(n) if allowed(out[i][j])]
            if vals:
                co[j] = min(vals)
                for i in range(n):
                    if allowed(out[i][j]):
                        out[i][j] = out[i][j] - co[j]
    return ReducedProblem(out, tuple(ro), tuple(co), sum(ro) + sum(co))


@dataclass(frozen=True)
class DulmageMendelsohn:
    """Coarse Dulmage-Mendelsohn data of the allowed-edge graph (unweighted)."""

    matching_size: int
    rows_missable: frozenset[int]  # rows left unmatched by at least one maximum matching
    cols_missable: frozenset[int]


def dulmage_mendelsohn(matrix: Any) -> DulmageMendelsohn:
    """Maximum matching size plus the rows / columns that some maximum matching leaves unmatched."""
    n, m, edges = _validated_edges(_as_lists(matrix))
    adj = [[j for j, _ in e] for e in edges]
    radj: list[list[int]] = [[] for _ in range(m)]
    for i, e in enumerate(adj):
        for j in e:
            radj[j].append(i)
    mr, mc = [-1] * n, [-1] * m

    def aug(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if mc[j] < 0 or aug(mc[j], seen):
                    mr[i], mc[j] = j, i
                    return True
        return False

    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10 * (n + m) + 1000))
    try:
        size = sum(aug(i, set()) for i in range(n))
    finally:
        sys.setrecursionlimit(old)

    def reach(free_rows: list[int], side_adj: list[list[int]], mate_other: list[int]) -> set[int]:
        seen_rows = set(free_rows)
        stack = list(free_rows)
        while stack:
            i = stack.pop()
            for j in side_adj[i]:
                k = mate_other[j]
                if k >= 0 and k not in seen_rows:
                    seen_rows.add(k)
                    stack.append(k)
        return seen_rows

    rows_missable = reach([i for i in range(n) if mr[i] < 0], adj, mc)
    cols_missable = reach([j for j in range(m) if mc[j] < 0], radj, mr)
    return DulmageMendelsohn(size, frozenset(rows_missable), frozenset(cols_missable))


def _fill_chain(
    i: int, c0: int, j: int, tight: Any, mp: Any, mate_r: Any, v: Any, on: int, om: int
) -> Any:
    """Rows to shift so that the vacated column `c0` is refilled, ending on a column with dual 0."""
    from collections import deque

    prev: dict[int, tuple[int, int, int] | None] = {c0: None}
    dq = deque([c0])
    while dq:
        c = dq.popleft()
        for r in range(on):
            if r == i or (r, c) not in tight or (r, c) in mp:
                continue
            c1 = mate_r[r]
            if c1 == j or c1 in prev:
                continue
            prev[c1] = (r, c, c)
            if v[c1] == 0:
                steps: list[tuple[str, tuple[int, int]]] = []
                cur = c1
                while (link := prev[cur]) is not None:
                    rr, cc, back = link
                    steps = [("remove", (rr, cur)), ("add", (rr, cc)), *steps]
                    cur = back
                return steps
            dq.append(c1)
    return None


def exchange_cycles(
    matrix: Any, maximize: bool = False
) -> list[tuple[tuple[str, tuple[int, int]], ...]]:
    """
    One alternating cycle (or alternating path ending on an unused column) for every optional,
    currently unmatched cell of the optimal assignment returned by `solve`. Each entry is a tuple of
    `("remove", cell)` / `("add", cell)` steps; applying it yields another optimal assignment.
    """
    from collections import deque

    from munkres._api import solve
    from munkres._verify import Certificate, _exact_rows, _orient, certify

    pairs = solve(matrix, maximize=maximize).pairs
    cert = certify(matrix, pairs, maximize)
    if not isinstance(cert, Certificate):
        raise AssertionError(cert.reason)
    n, m, rows = _exact_rows(matrix, maximize)
    on, om, orows, tr = _orient(n, m, rows)
    mp = {(c, r) if tr else (r, c) for r, c in pairs}
    mate_r = dict(mp)
    mate_c = {c: r for r, c in mp}
    tight = {
        (i, j) for i in range(on) for j, c in orows[i].items() if c - cert.u[i] - cert.v[j] == 0
    }
    flip = (lambda p: (p[1], p[0])) if tr else (lambda p: p)
    out: list[Any] = []
    for i, j in sorted(tight - mp):
        c0 = mate_r[i]
        steps: list[tuple[str, tuple[int, int]]] = [("remove", (i, c0)), ("add", (i, j))]
        r1 = mate_c.get(j)
        if r1 is None:  # j unused: row i moves there; c0 is vacated
            if (
                cert.v[c0] != 0
            ):  # a vacated column with non-zero dual must be refilled by another row
                fill = _fill_chain(i, c0, j, tight, mp, mate_r, cert.v, on, om)
                if fill is None:
                    continue
                steps += fill
            out.append(tuple((k, flip(p)) for k, p in steps))
            continue
        steps.append(("remove", (r1, j)))
        prev: dict[int, tuple[int, int] | None] = {r1: None}
        dq = deque([r1])
        end: tuple[int, int] | None = None
        while dq and end is None:
            r = dq.popleft()
            for c in range(om):
                if (r, c) not in tight or (r, c) in mp or c == j:
                    continue
                if c == c0 or (c not in mate_c and cert.v[c0] == 0):
                    end = (r, c)
                    break
                nxt = mate_c.get(c)
                if nxt is not None and nxt != i and nxt not in prev:
                    prev[nxt] = (r, c)
                    dq.append(nxt)
        if end is None:
            continue
        hops = [end]
        r = end[0]
        while (link := prev[r]) is not None:
            hops.append(link)
            r = link[0]
        for r, c in reversed(hops):
            steps.append(("add", (r, c)))
            if mate_c.get(c) is not None and c != c0:
                steps.append(("remove", (mate_c[c], c)))
        out.append(tuple((k, flip(p)) for k, p in steps))
    return out
