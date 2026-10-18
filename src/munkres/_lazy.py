# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Lazy / streaming costs with an exact optimality proof (no dense n x m cost matrix is ever stored).

`solve_lazy` solves on a sparse *candidate* graph, derives exact duals, then streams over every non-candidate
cell evaluating `cost(i, j)` once: any cell with negative reduced cost is added and the process repeats. When a full
pass finds none, the sparse optimum is a proven optimum of the full problem (dual feasibility on all cells), so a
bad candidate list can cost time but never correctness. Memory: O(candidate cells); time: O(n m) evaluations per pass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _Infeasible, _is_forbidden

__all__ = ["LazyResult", "solve_lazy", "solve_points"]


@dataclass(frozen=True)
class LazyResult:
    """Result of `solve_lazy`."""

    pairs: tuple[tuple[int, int], ...]
    total: Fraction
    candidate_cells: int
    evaluations: int
    rounds: int
    n_cells: int = 0

    @property
    def cells_avoided(self) -> int:
        """Cells whose cost was never evaluated."""
        return self.n_cells - self.evaluations


def _forbidden(v: Any) -> bool:
    return v is None or _is_forbidden(v)


def solve_lazy(
    n: int,
    m: int,
    cost: Callable[[int, int], Any],
    candidates: Callable[[int], Sequence[int]] | None = None,
    max_rounds: int = 200,
    allowed: Callable[[int, int], bool] | None = None,
    cost_rows: Callable[[int], Sequence[Any]] | None = None,
) -> LazyResult:
    """
    Exact minimum-cost assignment of all n rows (n <= m) with `cost(i, j)` evaluated on demand. `allowed(i, j)` is a predicate
    pushed down before any cost evaluation (cells it rejects are never evaluated); `cost_rows(i)` may supply a whole row at once.
    """
    from munkres._core import _solve
    from munkres._verify import _padded_duals

    if n > m:
        raise ValueError("requires n <= m")
    cache: dict[tuple[int, int], Any] = {}
    evals = 0

    def c(i: int, j: int) -> Any:
        nonlocal evals
        if (i, j) not in cache:
            if allowed is not None and not allowed(i, j):
                cache[(i, j)] = DISALLOWED
            elif cost_rows is not None:
                row = cost_rows(i)
                for jj in range(m):
                    if (i, jj) not in cache:
                        cache[(i, jj)] = (
                            DISALLOWED if allowed is not None and not allowed(i, jj) else row[jj]
                        )
                evals += m
            else:
                cache[(i, j)] = cost(i, j)
                evals += 1
        return cache[(i, j)]

    cand = [set(candidates(i)) if candidates else set(range(m)) for i in range(n)]
    for rnd in range(1, max_rounds + 1):
        rows = [
            {j: Fraction(c(i, j)) for j in sorted(cand[i]) if not _forbidden(c(i, j))}
            for i in range(n)
        ]
        try:
            pairs = _solve(n, m, [[(j, v) for j, v in r.items()] for r in rows])
        except (
            _Infeasible
        ) as bad:  # candidate graph has a Hall violation: open the offending rows completely
            for i in bad.rows:
                cand[i] = set(range(m))
            continue
        res = _padded_duals(n, m, rows, pairs)
        if not isinstance(res, tuple):
            raise AssertionError(res.reason)
        uu, vv, _, _, _ = res
        used = {j for _, j in pairs}
        free = [j for j in range(m) if j not in used]
        w = vv[free[0]] if free else Fraction(0)
        u = [uu[i] + w for i in range(n)]
        v = [vv[j] - w for j in range(m)]
        added = False
        for i in range(n):
            for j in range(m):
                if j not in cand[i]:
                    x = c(i, j)
                    if not _forbidden(x) and Fraction(x) - u[i] - v[j] < 0:
                        cand[i].add(j)
                        added = True
        if not added:
            total = sum((rows[i][j] for i, j in pairs), Fraction(0))
            return LazyResult(
                tuple(sorted(pairs)), total, sum(len(r) for r in rows), evals, rnd, n * m
            )
    raise RuntimeError("did not converge within max_rounds")


def solve_points(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], k: int = 8, squared: bool = True
) -> LazyResult:
    """
    Exact minimum-cost matching of points x into points y (rows <= cols) with Euclidean cost (squared by default),
    using each point's k nearest targets as the initial candidate graph and the exact dual check above.
    """

    def d(p: Sequence[float], q: Sequence[float]) -> float:
        return math.dist(p, q) ** 2 if squared else math.dist(p, q)

    near = [sorted(range(len(y)), key=lambda j: (d(x[i], y[j]), j))[:k] for i in range(len(x))]
    return solve_lazy(len(x), len(y), lambda i, j: d(x[i], y[j]), lambda i: near[i])
