# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Exact sensitivity and landscape tools built on duals/topology (small and medium problems)."""

from __future__ import annotations

import random as _random
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, UnsolvableMatrix, _as_lists, _validated_edges

__all__ = [
    "explain_edge",
    "ground_state_marginals",
    "hall_violator",
    "landscape_report",
    "near_optimal",
    "optimum_diameter",
    "sample_optimum",
    "vertex_indispensability",
]


def hall_violator(matrix: Any) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """
    For rows <= cols: `None` if every row can be matched, else an inclusion-minimal row set S with
    |N(S)| < |S| (Hall violator) and its neighbourhood N(S).
    """
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("hall_violator expects rows <= cols; pass the transpose")
    adj = [[j for j, _ in e] for e in edges]
    mc = [-1] * m
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
    sys.setrecursionlimit(max(old, 10 * (n + m) + 1000))
    try:
        for i in range(n):
            aug(i, set())
    finally:
        sys.setrecursionlimit(old)
    free = [i for i in range(n) if mr[i] < 0]
    if not free:
        return None
    rows, stack = {free[0]}, [free[0]]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            k = mc[j]
            if k >= 0 and k not in rows:
                rows.add(k)
                stack.append(k)

    def nbr(s: set[int]) -> set[int]:
        return {j for i in s for j in adj[i]}

    for i in sorted(rows):  # prune to an inclusion-minimal violator
        trial = rows - {i}
        if trial and len(nbr(trial)) < len(trial):
            rows = trial
    return tuple(sorted(rows)), tuple(sorted(nbr(rows)))


def optimum_diameter(matrix: Any, maximize: bool = False, limit: int = 2000) -> int:
    """Largest `matching_distance` between two optimal assignments (0 if the optimum is unique)."""
    from munkres._landscape import enumerate_optima, matching_distance

    sols = list(enumerate_optima(matrix, maximize, limit))
    return max((matching_distance(a, b) for a in sols for b in sols), default=0)


def vertex_indispensability(matrix: Any, maximize: bool = False) -> dict[str, dict[int, Any]]:
    """
    Exact objective change (caller's sense: new optimum - old optimum) if a single row / column were
    deleted from the problem; `None` if the problem becomes infeasible. Re-solves, so small problems.
    """
    from munkres._api import solve

    rows = _as_lists(matrix)
    n, m = len(rows), len(rows[0]) if rows else 0
    base = solve(rows, maximize=maximize).total

    def total(sub: list[list[Any]]) -> Any:
        if not sub or not sub[0]:
            return 0
        try:
            return solve(sub, maximize=maximize).total
        except UnsolvableMatrix:
            return None

    out: dict[str, dict[int, Any]] = {"rows": {}, "cols": {}}
    for i in range(n):
        t = total([r for k, r in enumerate(rows) if k != i])
        out["rows"][i] = None if t is None else t - base
    for j in range(m):
        t = total([[v for k, v in enumerate(r) if k != j] for r in rows])
        out["cols"][j] = None if t is None else t - base
    return out


def explain_edge(matrix: Any, i: int, j: int, maximize: bool = False) -> str:
    """Plain-language, exact explanation of why cell (i, j) is / is not in the optimal solution(s)."""
    from munkres._topology import edge_margins, topology

    t = topology(matrix, maximize=maximize)
    cell = (i, j)
    if cell not in t.reduced_cost:
        return f"cell {cell} is forbidden (or out of range): it can never be used."
    status = (
        "forced (in every optimal assignment)"
        if cell in t.forced
        else (
            "optional (in some optimal assignments, not all)"
            if cell in t.optional
            else "not in any optimal assignment"
        )
    )
    mg = edge_margins(matrix, maximize)[cell]
    ex, fr = mg["exclude_penalty"], mg["force_penalty"]
    parts = [f"cell {cell} is {status}.", f"Reduced cost {t.reduced_cost[cell]} (0 means tight)."]
    parts.append(
        "Excluding it makes the problem infeasible."
        if ex is None
        else f"Forbidding it worsens the optimum by {ex}."
    )
    parts.append(
        "Forcing it is infeasible." if fr is None else f"Forcing it worsens the optimum by {fr}."
    )
    return " ".join(parts)


def near_optimal(
    matrix: Any, eps: Any, maximize: bool = False, limit: int = 100_000
) -> list[tuple[Any, tuple[tuple[int, int], ...]]]:
    """
    Every assignment whose objective is within `eps` of the optimum (worse direction), as
    `(gap, pairs)` sorted by gap. Exact: gap = sum of reduced costs of used cells + sum of -v over
    unused columns, pruned by `eps`. Raises `RuntimeError` past `limit` results.
    """
    from munkres._api import solve
    from munkres._verify import Certificate, _exact_rows, _orient, certify

    pairs = solve(matrix, maximize=maximize).pairs
    cert = certify(matrix, pairs, maximize)
    if not isinstance(cert, Certificate):
        raise AssertionError(cert.reason)
    n, m, rows = _exact_rows(matrix, maximize)
    on, om, orows, tr = _orient(n, m, rows)
    eps = Fraction(eps)
    red = [{j: c - cert.u[i] - cert.v[j] for j, c in orows[i].items()} for i in range(on)]
    out: list[Any] = []

    def rec(i: int, used: int, gap: Fraction, acc: list[tuple[int, int]]) -> None:
        if gap > eps:
            return
        if i == on:
            unused = sum((-cert.v[j] for j in range(om) if not used >> j & 1), Fraction(0))
            if gap + unused <= eps:
                if len(out) >= limit:
                    raise RuntimeError(f"more than {limit} near-optimal assignments")
                out.append((gap + unused, tuple(sorted((c, r) if tr else (r, c) for r, c in acc))))
            return
        for j, r in red[i].items():
            if not used >> j & 1:
                acc.append((i, j))
                rec(i + 1, used | 1 << j, gap + r, acc)
                acc.pop()

    rec(0, 0, Fraction(0), [])
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def sample_optimum(
    matrix: Any, rng: Any = None, maximize: bool = False, limit: int = 100_000
) -> tuple[tuple[int, int], ...]:
    """Uniformly random optimal assignment (exact: enumerates the optima, so small/medium problems)."""
    from munkres._landscape import enumerate_optima

    rng = rng or _random.Random()  # noqa: S311  # nosec B311
    picked: tuple[tuple[int, int], ...] = rng.choice(
        list(enumerate_optima(matrix, maximize, limit))
    )
    return picked


def ground_state_marginals(
    matrix: Any, maximize: bool = False, limit: int = 100_000
) -> dict[tuple[int, int], Fraction]:
    """Fraction of optimal assignments using each cell = the zero-temperature (beta -> infinity) Gibbs marginals."""
    from munkres._landscape import enumerate_optima

    sols = list(enumerate_optima(matrix, maximize, limit))
    out: dict[tuple[int, int], Fraction] = {}
    for s in sols:
        for cell in s:
            out[cell] = out.get(cell, Fraction(0)) + Fraction(1, len(sols))
    return out


def landscape_report(matrix: Any, beta: float = 1.0, maximize: bool = False) -> dict[str, Any]:
    """One call summarising optimum, number of optima, face dimension, feasible-assignment count and Gibbs statistics."""
    from munkres._api import solve
    from munkres._gibbs import count_matchings, gibbs
    from munkres._landscape import face_dimension
    from munkres._topology import topology

    rows = _as_lists(matrix)
    top = topology(rows, maximize=maximize)
    sign = -1 if maximize else 1
    g = gibbs([[DISALLOWED if v is DISALLOWED else sign * v for v in r] for r in rows], beta)
    return {
        "optimum": solve(rows, maximize=maximize).total,
        "n_optimal": top.n_optimal,
        "unique": top.unique,
        "face_dimension": face_dimension(rows, maximize)
        if top.n_optimal is not None and top.n_optimal <= 2000
        else None,
        "n_feasible_assignments": count_matchings(rows),
        "gibbs_entropy": g.entropy,
        "gibbs_log_z": g.log_z,
        "forced": sorted(top.forced),
        "optional": sorted(top.optional),
    }
