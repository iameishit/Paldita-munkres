# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Algebraic matching backends (independent of augmenting paths) and permanent tools.

* `has_perfect_matching_algebraic`: Edmonds' theorem. A random weighting of the allowed cells gives a
  non-zero determinant mod p iff (w.h.p.) a perfect matching exists. One-sided: True is certain, False
  is wrong with probability <= n/p (Schwartz-Zippel).
* `algebraic_optimum`: min total weight over perfect matchings of non-negative integer weights, read off the lowest
  non-zero coefficient of det[ r_ij x^{w_ij} ] (found by interpolation). Same one-sided error bound.
* `permanent` (exact subset DP) and `permanent_bounds` (Sinkhorn scaling + van der Waerden / Egorychev-Falikman).
"""

from __future__ import annotations

import math
import random as _random
from typing import Any

from munkres._core import _as_lists, _validated_edges

__all__ = ["algebraic_optimum", "has_perfect_matching_algebraic", "permanent", "permanent_bounds"]

_P = (1 << 61) - 1


def _det_mod(m: list[list[int]], p: int) -> int:
    n = len(m)
    a = [row[:] for row in m]
    det = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] % p), None)
        if piv is None:
            return 0
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            det = -det
        det = det * a[c][c] % p
        inv = pow(a[c][c], -1, p)
        for r in range(c + 1, n):
            f = a[r][c] * inv % p
            if f:
                a[r] = [(x - f * y) % p for x, y in zip(a[r], a[c], strict=False)]
    return det % p


def _square_pattern(matrix: Any) -> tuple[int, list[list[Any]]]:
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n != m:
        raise ValueError("square matrix required")
    return n, edges


def has_perfect_matching_algebraic(matrix: Any, rng: Any = None, p: int = _P) -> bool:
    """Randomised existence test for a perfect matching of the allowed cells (see module notes for the error)."""
    n, edges = _square_pattern(matrix)
    rng = rng or _random.Random()  # noqa: S311  # nosec B311
    m = [[0] * n for _ in range(n)]
    for i, e in enumerate(edges):
        for j, _ in e:
            m[i][j] = rng.randrange(1, p)
    return n == 0 or _det_mod(m, p) != 0


def algebraic_optimum(
    matrix: Any, maximize: bool = False, rng: Any = None, p: int = _P
) -> int | None:
    """
    Optimum of a square matrix with non-negative integer weights, computed only from determinants mod p.
    Returns None if no perfect matching exists. Error probability <= n/p (the result can only be too large).
    """
    n, edges = _square_pattern(matrix)
    w = [dict(e) for e in edges]
    if any(not isinstance(c, int) or c < 0 for r in w for c in r.values()):
        raise ValueError("weights must be non-negative integers")
    if n == 0:
        return 0
    top = max((c for r in w for c in r.values()), default=0)
    if maximize:
        w = [{j: top - c for j, c in r.items()} for r in w]
    rng = rng or _random.Random()  # noqa: S311  # nosec B311
    r = [{j: rng.randrange(1, p) for j in row} for row in w]
    deg = n * top
    xs = list(range(1, deg + 2))
    ys = []
    for x in xs:
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j, c in w[i].items():
                m[i][j] = r[i][j] * pow(x, c, p) % p
        ys.append(_det_mod(m, p))
    master = [1]
    for xs_ in xs:  # master polynomial prod (X - xs_), coefficients low -> high
        nxt = [0] * (len(master) + 1)
        for k, c in enumerate(master):
            nxt[k + 1] = (nxt[k + 1] + c) % p
            nxt[k] = (nxt[k] - xs_ * c) % p
        master = nxt
    coef = [0] * (deg + 1)
    for t, x in enumerate(xs):
        q, carry = [0] * (len(master) - 1), 0  # synthetic division master / (X - x)
        for k in range(len(master) - 1, 0, -1):
            carry = (master[k] + carry * x) % p
            q[k - 1] = carry
        denom = 1
        for s, x2 in enumerate(xs):
            if s != t:
                denom = denom * (x - x2) % p
        scale = ys[t] * pow(denom, -1, p) % p
        for k in range(deg + 1):
            coef[k] = (coef[k] + scale * q[k]) % p
    low = next((k for k, c in enumerate(coef) if c), None)
    if low is None:
        return None
    return n * top - low if maximize else low


def permanent(matrix: Any) -> Any:
    """Exact permanent (sum over injections of all rows into distinct columns), subset DP, at most 20 columns."""
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("requires rows <= cols")
    if m > 20:
        raise ValueError("at most 20 columns")
    dp: dict[int, Any] = {0: 1}
    for i in range(n):
        nxt: dict[int, Any] = {}
        for mask, v in dp.items():
            for j, c in edges[i]:
                if not mask >> j & 1:
                    nxt[mask | 1 << j] = nxt.get(mask | 1 << j, 0) + v * c
        dp = nxt
    return sum(dp.values()) if dp else 0


def permanent_bounds(matrix: Any, iters: int = 2000) -> tuple[float, float]:
    """
    (lower, upper) bounds on the permanent of a square non-negative matrix with a positive diagonal pattern:
    per(A) = per(D) / (prod r prod c) for the Sinkhorn-scaled doubly stochastic D = R A C, and
    n!/n^n <= per(D) <= 1. Accurate to the Sinkhorn convergence (floating point), not a proof for rounded input.
    """
    a = [[float(v) for v in r] for r in _as_lists(matrix)]
    n = len(a)
    if n == 0 or any(len(r) != n for r in a) or any(v < 0 for r in a for v in r):
        raise ValueError("square non-negative matrix required")
    r, c = [1.0] * n, [1.0] * n
    for _ in range(iters):
        for i in range(n):
            s = sum(a[i][j] * c[j] for j in range(n))
            r[i] = 1.0 / s
        for j in range(n):
            s = sum(a[i][j] * r[i] for i in range(n))
            c[j] = 1.0 / s
    scale = math.prod(r) * math.prod(c)
    return math.factorial(n) / n**n / scale, 1.0 / scale
