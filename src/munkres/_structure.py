# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Structure detection ("mathematical microscope") and structure-specific solvers.

* `fingerprint` - cheap, deterministic structural summary of a cost matrix.
* `monge_witness` / `inverse_monge_witness` - exact tests with a violating 2x2
  witness; a `None` result is a *proof* of the property (checked on exact rationals).
* `solve_monge` - O(n*m) dynamic program, valid only when the matrix is Monge
  (or inverse Monge); falls back to the general solver otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _is_nonfinite, _safe_ratio, _validated_edges

__all__ = [
    "Fingerprint",
    "fingerprint",
    "inverse_monge_witness",
    "is_additive",
    "monge_witness",
    "solve_monge",
]

_INF = float("inf")


def _dense_exact(matrix: Any) -> list[list[Fraction]] | None:
    """Exact rational copy if every cell is finite, else None (forbidden cells break Monge)."""
    rows = _as_lists(matrix)
    out = []
    for row in rows:
        r = []
        for v in row:
            if v is DISALLOWED or _is_nonfinite(v):
                return None
            r.append(Fraction(v))
        out.append(r)
    return out


def monge_witness(matrix: Any) -> tuple[int, int] | None:
    """
    `None` iff `matrix` is Monge: c[i][j] + c[i+1][j+1] <= c[i][j+1] + c[i+1][j]
    for all i, j (adjacent minors suffice). Otherwise `(i, j)` of a violating 2x2.
    Matrices with forbidden cells are reported as violating at the first one found
    adjacent to a finite cell, since the inequality is undefined there.
    """
    a = _dense_exact(matrix)
    if a is None:
        return (0, 0)
    for i in range(len(a) - 1):
        for j in range(len(a[0]) - 1):
            if a[i][j] + a[i + 1][j + 1] > a[i][j + 1] + a[i + 1][j]:
                return (i, j)
    return None


def inverse_monge_witness(matrix: Any) -> tuple[int, int] | None:
    """Like `monge_witness` for the reversed inequality (c[i][j]+c[i+1][j+1] >= ...)."""
    a = _dense_exact(matrix)
    if a is None:
        return (0, 0)
    for i in range(len(a) - 1):
        for j in range(len(a[0]) - 1):
            if a[i][j] + a[i + 1][j + 1] < a[i][j + 1] + a[i + 1][j]:
                return (i, j)
    return None


def is_additive(matrix: Any) -> bool:
    """True iff c[i][j] = a[i] + b[j] (both Monge and inverse Monge): every assignment is optimal."""
    return monge_witness(matrix) is None and inverse_monge_witness(matrix) is None


def _monotone_dp(a: list[list[Fraction]]) -> list[tuple[int, int]]:
    n, m = len(a), len(a[0])
    dp: list[list[Fraction | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = Fraction(0)
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            best, took = dp[i][j - 1], False
            prev = dp[i - 1][j - 1]
            if prev is not None:
                cand = prev + a[i - 1][j - 1]
                if best is None or cand < best:
                    best, took = cand, True
            dp[i][j], take[i][j] = best, took
    out, j = [], m
    for i in range(n, 0, -1):
        while not take[i][j]:
            j -= 1
        out.append((i - 1, j - 1))
        j -= 1
    return out[::-1]


def solve_monge(matrix: Any, maximize: bool = False) -> tuple[list[tuple[int, int]], str]:
    """
    Optimal assignment of min(n, m) cardinality using the Monge structure when it
    is *proved* present; otherwise the general solver. Returns `(pairs, method)`
    with method in {"monge-dp", "inverse-monge-dp", "general"}.
    """
    from munkres._api import _negated, solve

    base = _negated(_as_lists(matrix)) if maximize else _as_lists(matrix)
    n = len(base)
    m = len(base[0]) if n else 0
    if n and m and n <= m:
        a = _dense_exact(base)
        if a is not None:
            if monge_witness(base) is None:
                return _monotone_dp(a), "monge-dp"
            if inverse_monge_witness(base) is None:
                rev = [row[::-1] for row in a]
                return sorted((i, m - 1 - j) for i, j in _monotone_dp(rev)), "inverse-monge-dp"
    return list(solve(matrix, maximize=maximize).pairs), "general"


@dataclass(frozen=True)
class Fingerprint:
    """Deterministic structural summary. Every field is a plain fact, not a prediction."""

    shape: tuple[int, int]
    allowed_cells: int
    density: float  # allowed / total
    min_cost: Any
    max_cost: Any
    dynamic_range: float | None  # max |c| / min nonzero |c| among allowed cells
    duplicate_rows: int
    duplicate_columns: int
    row_tie_fraction: float  # rows whose minimum is attained more than once
    components: int  # connected components of the allowed-edge graph (rows+cols)
    isolated_rows: int
    isolated_columns: int
    monge: bool
    inverse_monge: bool
    additive: bool
    notes: tuple[str, ...] = field(default=())


def fingerprint(matrix: Any) -> Fingerprint:
    """Compute a `Fingerprint`. O(n*m) time; exact for Monge/additive claims."""
    rows = _as_lists(matrix)
    n, m, edges = _validated_edges(rows)
    allowed = sum(len(e) for e in edges)
    vals = [c for e in edges for _, c in e]
    nonzero = [abs(c) for c in vals if c != 0]
    dyn = _safe_ratio(max(nonzero), min(nonzero)) if nonzero else None
    row_sig = [tuple(sorted(e)) for e in edges]
    col_edges: list[list[tuple[int, Any]]] = [[] for _ in range(m)]
    for i, e in enumerate(edges):
        for j, c in e:
            col_edges[j].append((i, c))
    col_sig = [tuple(sorted(c)) for c in col_edges]
    ties = 0
    for e in edges:
        if e:
            lo = min(c for _, c in e)
            ties += sum(1 for _, c in e if c == lo) > 1
    parent = list(range(n + m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, e in enumerate(edges):
        for j, _ in e:
            parent[find(i)] = find(n + j)
    comps = len({find(x) for x in range(n + m)})
    mw = monge_witness(rows) is None if n and m else True
    imw = inverse_monge_witness(rows) is None if n and m else True
    notes = []
    if allowed < n * m:
        notes.append("has forbidden cells: Monge tests need a fully finite matrix")
    if dyn is not None and dyn > 1e12:
        notes.append("very large dynamic range: float arithmetic may lose precision")
    return Fingerprint(
        (n, m),
        allowed,
        allowed / (n * m) if n * m else 1.0,
        min(vals) if vals else None,
        max(vals) if vals else None,
        dyn,
        n - len(set(row_sig)) if n else 0,
        m - len(set(col_sig)) if m else 0,
        ties / n if n else 0.0,
        comps,
        sum(1 for e in edges if not e),
        sum(1 for c in col_edges if not c),
        mw and allowed == n * m,
        imw and allowed == n * m,
        mw and imw and allowed == n * m,
        tuple(notes),
    )
