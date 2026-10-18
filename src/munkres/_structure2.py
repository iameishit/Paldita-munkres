# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""More exact structure detectors, semiring permanents and the fine Dulmage-Mendelsohn decomposition."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _is_nonfinite, _validated_edges

__all__ = [
    "SEMIRINGS",
    "DMBlocks",
    "Semiring",
    "dm_blocks",
    "is_circulant",
    "is_toeplitz",
    "is_totally_monotone",
    "is_totally_unimodular",
    "is_tree_metric",
    "is_ultrametric",
    "kronecker_factors",
    "leftmost_row_minima",
    "monge_violations",
    "row_minima_monotone",
    "semiring_permanent",
    "tropical_rank",
]

_INF = float("inf")


def _finite_fractions(matrix: Any, who: str) -> list[list[Fraction]]:
    """Exact rational copy of `matrix`; NaN/infinite cells raise a clear `ValueError` (never OverflowError)."""
    out: list[list[Fraction]] = []
    for i, row in enumerate(_as_lists(matrix)):
        r: list[Fraction] = []
        for j, v in enumerate(row):
            if _is_nonfinite(v):
                raise ValueError(f"{who} needs finite costs; cell [{i}][{j}] is {v!r}")
            r.append(Fraction(v))
        out.append(r)
    return out


def _det(a: list[list[Fraction]]) -> Fraction:
    n = len(a)
    a = [r[:] for r in a]
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            if f:
                a[r] = [x - f * y for x, y in zip(a[r], a[c], strict=False)]
    return det


def is_totally_unimodular(matrix: Any, max_dim: int = 8) -> bool:
    """True iff every entry is in {-1, 0, 1} and every square submatrix has determinant in {-1, 0, 1} (exact; both dims <= max_dim)."""
    a = _finite_fractions(matrix, "is_totally_unimodular")
    n, m = len(a), len(a[0]) if a else 0
    if n > max_dim or m > max_dim:
        raise ValueError(f"exhaustive check limited to {max_dim} x {max_dim}")
    if any(v not in (-1, 0, 1) for r in a for v in r):
        return False
    for k in range(2, min(n, m) + 1):
        for rs in itertools.combinations(range(n), k):
            for cs in itertools.combinations(range(m), k):
                if _det([[a[i][j] for j in cs] for i in rs]) not in (-1, 0, 1):
                    return False
    return True


def kronecker_factors(matrix: Any) -> list[tuple[list[list[Fraction]], list[list[Fraction]]]]:
    """Every exact factorisation C = A (x) B with non-trivial block shapes (A p x q, B (n/p) x (m/q)); B is normalised so its first non-zero is 1... up to a common scalar."""
    c = _finite_fractions(matrix, "kronecker_factors")
    n, m = len(c), len(c[0]) if c else 0
    out = []
    for p in range(1, n + 1):
        for q in range(1, m + 1):
            if n % p or m % q or (p, q) in ((1, 1), (n, m)):
                continue
            r, s = n // p, m // q
            blocks = {
                (i, j): [[c[i * r + x][j * s + y] for y in range(s)] for x in range(r)]
                for i in range(p)
                for j in range(q)
            }
            ref = next((b for b in blocks.values() if any(v != 0 for row in b for v in row)), None)
            if ref is None:
                continue
            px, py = next((x, y) for x in range(r) for y in range(s) if ref[x][y] != 0)
            a = [[blocks[(i, j)][px][py] / ref[px][py] for j in range(q)] for i in range(p)]
            if all(
                blocks[(i, j)][x][y] == a[i][j] * ref[x][y]
                for i in range(p)
                for j in range(q)
                for x in range(r)
                for y in range(s)
            ):
                out.append((a, [row[:] for row in ref]))
    return out


def is_toeplitz(matrix: Any) -> bool:
    """c[i][j] == c[i-1][j-1] for all i, j >= 1."""
    a = _as_lists(matrix)
    return all(a[i][j] == a[i - 1][j - 1] for i in range(1, len(a)) for j in range(1, len(a[0])))


def is_circulant(matrix: Any) -> bool:
    """Square and every row is the previous row rotated right by one."""
    a = _as_lists(matrix)
    n = len(a)
    return all(len(r) == n for r in a) and all(
        a[i] == a[i - 1][-1:] + a[i - 1][:-1] for i in range(1, n)
    )


def _metric_ok(d: list[list[Any]]) -> bool:
    n = len(d)
    return (
        all(len(r) == n for r in d)
        and all(d[i][i] == 0 for i in range(n))
        and all(d[i][j] == d[j][i] for i in range(n) for j in range(n))
    )


def is_ultrametric(dist: Any) -> bool:
    """Symmetric, zero diagonal and d(x,z) <= max(d(x,y), d(y,z)) for all triples."""
    d = _as_lists(dist)
    n = len(d)
    return _metric_ok(d) and all(
        d[x][z] <= max(d[x][y], d[y][z]) for x in range(n) for y in range(n) for z in range(n)
    )


def is_tree_metric(dist: Any) -> bool:
    """Metric satisfying the four-point condition: of the three pair sums for any 4 points, the two largest are equal."""
    d = _as_lists(dist)
    n = len(d)
    if not _metric_ok(d) or any(
        d[x][z] > d[x][y] + d[y][z] for x in range(n) for y in range(n) for z in range(n)
    ):
        return False
    for w, x, y, z in itertools.combinations(range(n), 4):
        s = sorted([d[w][x] + d[y][z], d[w][y] + d[x][z], d[w][z] + d[x][y]])
        if s[1] != s[2]:
            return False
    return True


def monge_violations(matrix: Any) -> dict[str, Any]:
    """Near-Monge diagnostics: the adjacent 2x2 minors that violate the Monge inequality, with their excess, and the violating fraction."""
    a = _finite_fractions(matrix, "monge_violations")
    bad = []
    total = 0
    for i in range(len(a) - 1):
        for j in range(len(a[0]) - 1):
            total += 1
            ex = a[i][j] + a[i + 1][j + 1] - a[i][j + 1] - a[i + 1][j]
            if ex > 0:
                bad.append((i, j, ex))
    return {
        "violations": bad,
        "minors": total,
        "fraction": Fraction(len(bad), total) if total else Fraction(0),
    }


def leftmost_row_minima(matrix: Any) -> list[int]:
    """Column of the leftmost minimum of every row (brute force, O(nm))."""
    a = _as_lists(matrix)
    return [min(range(len(r)), key=lambda j: (r[j], j)) for r in a]


def is_totally_monotone(matrix: Any) -> bool:
    """Every 2x2 submatrix (rows i<k, columns j<l) with a[i][j] > a[i][l] also has a[k][j] > a[k][l] (row-minima monotone)."""
    a = _as_lists(matrix)
    n, m = len(a), len(a[0]) if a else 0
    return all(
        not (a[i][j] > a[i][col2]) or a[k][j] > a[k][col2]
        for i in range(n)
        for k in range(i + 1, n)
        for j in range(m)
        for col2 in range(j + 1, m)
    )


def row_minima_monotone(matrix: Any, check: bool = True) -> list[int]:
    """Leftmost row minima of a totally monotone matrix by divide and conquer (O((n + m) log n) evaluations)."""
    a = _as_lists(matrix)
    if check and not is_totally_monotone(a):
        raise ValueError("matrix is not totally monotone")
    n, m = len(a), len(a[0]) if a else 0
    out = [0] * n

    def rec(r0: int, r1: int, c0: int, c1: int) -> None:
        if r0 > r1:
            return
        mid = (r0 + r1) // 2
        best = min(range(c0, c1 + 1), key=lambda j: (a[mid][j], j))
        out[mid] = best
        rec(r0, mid - 1, c0, best)
        rec(mid + 1, r1, best, c1)

    rec(0, n - 1, 0, m - 1)
    return out


def tropical_rank(matrix: Any) -> int:
    """Largest k such that some k x k minor has a *unique* minimum-cost assignment (min-plus non-singular); at most 6 x 6."""
    a = _as_lists(matrix)
    n, m = len(a), len(a[0]) if a else 0
    if n > 6 or m > 6:
        raise ValueError("limited to 6 x 6")
    for k in range(min(n, m), 0, -1):
        for rs in itertools.combinations(range(n), k):
            for cs in itertools.combinations(range(m), k):
                vals = []
                for p in itertools.permutations(cs):
                    cells = [a[r][c] for r, c in zip(rs, p, strict=False)]
                    if all(v is not DISALLOWED and v != _INF for v in cells):
                        vals.append(sum(cells))
                if vals and vals.count(min(vals)) == 1:
                    return k
    return 0


@dataclass(frozen=True)
class Semiring:
    """(add, mul, zero, one) for `semiring_permanent`."""

    add: Callable[[Any, Any], Any]
    mul: Callable[[Any, Any], Any]
    zero: Any
    one: Any


SEMIRINGS = {
    "min_plus": Semiring(min, lambda x, y: x + y, _INF, 0),
    "max_plus": Semiring(max, lambda x, y: x + y, -_INF, 0),
    "counting": Semiring(lambda x, y: x + y, lambda x, y: x * y, 0, 1),
    "boolean": Semiring(lambda x, y: x or y, lambda x, y: x and y, False, True),
    "max_times": Semiring(max, lambda x, y: x * y, 0, 1),
}


def semiring_permanent(matrix: Any, semiring: str | Semiring = "min_plus") -> Any:
    """
    Sum (semiring `add`) over injections of all rows into distinct columns of the product (semiring `mul`) of the chosen cells.
    min_plus = assignment optimum, counting = permanent, boolean = a perfect matching exists. Forbidden cells are skipped; <= 20 columns.
    """
    sr = SEMIRINGS[semiring] if isinstance(semiring, str) else semiring
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("requires rows <= cols")
    if m > 20:
        raise ValueError("at most 20 columns")
    dp: dict[int, Any] = {0: sr.one}
    for i in range(n):
        nxt: dict[int, Any] = {}
        for mask, v in dp.items():
            for j, c in edges[i]:
                if not mask >> j & 1:
                    k, val = mask | 1 << j, sr.mul(v, c)
                    nxt[k] = sr.add(nxt[k], val) if k in nxt else val
        dp = nxt
    out = sr.zero
    for v in dp.values():
        out = sr.add(out, v)
    return out


@dataclass(frozen=True)
class DMBlocks:
    """Fine Dulmage-Mendelsohn blocks of a square matrix with a perfect matching, and the cells used by some perfect matching."""

    blocks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    allowed_cells: frozenset[tuple[int, int]]


def dm_blocks(matrix: Any) -> DMBlocks:
    """Strongly connected blocks of the allowed-edge graph under a perfect matching; a cell is in some perfect matching iff its endpoints share a block."""
    from munkres._topology import _scc

    n, m, edges = _validated_edges(_as_lists(matrix))
    if n != m:
        raise ValueError("square matrix required")
    adj = [[j for j, _ in e] for e in edges]
    mc = [-1] * n
    mr = [-1] * n

    def aug(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if mc[j] < 0 or aug(mc[j], seen):
                    mc[j], mr[i] = i, j
                    return True
        return False

    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10 * n + 1000))
    try:
        ok = all(aug(i, set()) for i in range(n))
    finally:
        sys.setrecursionlimit(old)
    if not ok:
        raise ValueError("no perfect matching")
    g: list[list[int]] = [[] for _ in range(2 * n)]
    for i in range(n):
        for j in adj[i]:
            if mr[i] == j:
                g[n + j].append(i)
            else:
                g[i].append(n + j)
    comp = _scc(2 * n, g)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for v in range(2 * n):
        grp = groups.setdefault(comp[v], ([], []))
        (grp[0] if v < n else grp[1]).append(v if v < n else v - n)
    allowed = {(i, j) for i in range(n) for j in adj[i] if mr[i] == j or comp[i] == comp[n + j]}
    return DMBlocks(
        tuple((tuple(sorted(r)), tuple(sorted(c))) for r, c in sorted(groups.values())),
        frozenset(allowed),
    )
