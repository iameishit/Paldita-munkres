# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Risk-aware and 3-index assignment, exact by enumeration / subset DP (small problems)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists, _is_forbidden, _is_nonfinite

__all__ = ["cvar_assignment", "mean_variance_assignment", "solve_3d"]


def _scenario_costs(scenarios: Sequence[Any], max_n: int) -> tuple[list[list[list[Any]]], int, int]:
    mats = [_as_lists(s) for s in scenarios]
    n, m = len(mats[0]), len(mats[0][0])
    if n > m or n > max_n:
        raise ValueError(f"supports rows <= cols and n <= {max_n} (enumeration)")
    return mats, n, m


def _enumerate(mats: list[Any], n: int, m: int) -> Iterator[tuple[tuple[int, ...], list[Fraction]]]:
    for cols in itertools.permutations(range(m), n):
        if all(s[i][c] is not DISALLOWED for s in mats for i, c in enumerate(cols)):
            yield (
                cols,
                [sum((Fraction(s[i][c]) for i, c in enumerate(cols)), Fraction(0)) for s in mats],
            )


def cvar_assignment(
    scenarios: Sequence[Any], alpha: float, max_n: int = 8
) -> tuple[tuple[tuple[int, int], ...], Fraction]:
    """
    Assignment minimising CVaR_alpha of the cost over equally likely scenarios: the mean of the worst
    ceil((1 - alpha) * S) scenario costs. alpha = 0 is the expectation; alpha -> 1 is the worst case.
    """
    mats, n, m = _scenario_costs(scenarios, max_n)
    k = max(1, math.ceil((1 - alpha) * len(mats) - 1e-12))
    best: tuple[tuple[tuple[int, int], ...], Fraction] | None = None
    for cols, costs in _enumerate(mats, n, m):
        v = sum(sorted(costs, reverse=True)[:k], Fraction(0)) / k
        if best is None or v < best[1]:
            best = (tuple(enumerate(cols)), v)
    if best is None:
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible assignment")
    return best


def mean_variance_assignment(
    scenarios: Sequence[Any], lam: Any, max_n: int = 8
) -> tuple[tuple[tuple[int, int], ...], Fraction]:
    """Assignment minimising mean + lam * (population) variance of the cost over equally likely scenarios."""
    mats, n, m = _scenario_costs(scenarios, max_n)
    lam, s = Fraction(lam), len(mats)
    best: tuple[tuple[tuple[int, int], ...], Fraction] | None = None
    for cols, costs in _enumerate(mats, n, m):
        mu = sum(costs, Fraction(0)) / s
        var = sum(((c - mu) ** 2 for c in costs), Fraction(0)) / s
        v = mu + lam * var
        if best is None or v < best[1]:
            best = (tuple(enumerate(cols)), v)
    if best is None:
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible assignment")
    return best


def solve_3d(cost: Any, maximize: bool = False) -> tuple[tuple[tuple[int, int, int], ...], Any]:
    """
    Axial 3-index assignment: choose one cell (i, j, k) per i with all j distinct and all k distinct, minimising
    (or maximising) total cost. NP-hard in general; exact subset DP over (used j, used k), so n <= 8 and
    dims_j, dims_k <= 10. Forbidden cells: DISALLOWED. Returns `(triples, total)`.
    """
    n, nj, nk = len(cost), len(cost[0]), len(cost[0][0])
    if n > 8 or nj > 10 or nk > 10:
        raise ValueError("exact DP limited to n <= 8 and j, k <= 10")
    pick = max if maximize else min
    dp: dict[tuple[int, int], tuple[Any, tuple[Any, ...]]] = {(0, 0): (0, ())}
    for i in range(n):
        nxt: dict[tuple[int, int], tuple[Any, tuple[Any, ...]]] = {}
        for (mj, mk), (v, path) in dp.items():
            for j in range(nj):
                if mj >> j & 1:
                    continue
                for k in range(nk):
                    c = cost[i][j][k]
                    if mk >> k & 1 or _is_forbidden(c) or _is_nonfinite(c):
                        continue
                    key = (mj | 1 << j, mk | 1 << k)
                    cand = (v + c, (*path, (i, j, k)))
                    if key not in nxt or (
                        pick(cand[0], nxt[key][0]) == cand[0] and cand[0] != nxt[key][0]
                    ):
                        nxt[key] = cand
        dp = nxt
    if not dp:
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible 3-index assignment")
    best = pick(dp.values(), key=lambda t: t[0])
    return best[1], best[0]
