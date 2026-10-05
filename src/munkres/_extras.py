# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Related matching problems: stable matching, transportation, entropic transport."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from munkres._api import solve
from munkres._core import MatrixLike, UnsolvableMatrix, _as_lists, _validated_edges

__all__ = [
    "SinkhornResult",
    "Transport",
    "sinkhorn",
    "soft_assignment",
    "stable_matching",
    "transport",
]


# ---------------------------------------------------------------------------
# Gale-Shapley stable matching
# ---------------------------------------------------------------------------


def stable_matching(
    proposers: Mapping[Any, Sequence[Any]], receivers: Mapping[Any, Sequence[Any]]
) -> dict[Any, Any]:
    """
    Stable matching by deferred acceptance (Gale-Shapley), for problems where
    people have *preferences* instead of costs (students/schools, residents/
    hospitals, ...). Each argument maps a name to the partners it finds
    acceptable, most preferred first. Unacceptable pairings are simply not listed
    (a pair is only made if *both* list each other).

    Returns `{proposer: receiver}` for the matched proposers. The result is
    stable (no two parties would both rather be together than with their
    partners) and **proposer-optimal**: every proposer gets the best partner any
    stable matching can give them. Swap the arguments to favour the receivers.
    """
    for side, prefs, other in (
        ("proposer", proposers, receivers),
        ("receiver", receivers, proposers),
    ):
        for name, ranked in prefs.items():
            if len(set(ranked)) != len(ranked):
                raise ValueError(f"{side} {name!r} lists someone twice")
            unknown = [x for x in ranked if x not in other]
            if unknown:
                raise ValueError(f"{side} {name!r} lists unknown partner(s) {unknown!r}")
    rank = {r: {p: k for k, p in enumerate(ranked)} for r, ranked in receivers.items()}
    next_choice = dict.fromkeys(proposers, 0)
    held: dict[Any, Any] = {}
    free = list(proposers)
    while free:
        p = free.pop()
        ranked = proposers[p]
        while next_choice[p] < len(ranked):
            r = ranked[next_choice[p]]
            next_choice[p] += 1
            if p not in rank[r]:
                continue
            current = held.get(r)
            if current is None:
                held[r] = p
                break
            if rank[r][p] < rank[r][current]:
                held[r] = p
                free.append(current)
                break
    matched = {p: r for r, p in held.items()}
    return {p: matched[p] for p in proposers if p in matched}


# ---------------------------------------------------------------------------
# Transportation problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transport:
    """Result of `transport`: `flows[(supplier, consumer)] = units`."""

    flows: dict[tuple[int, int], int]
    total_cost: Any
    shipped: int


def transport(
    supply: Sequence[int],
    demand: Sequence[int],
    cost: MatrixLike,
    *,
    max_units: int = 1000,
) -> Transport:
    """
    The transportation problem: ship goods from suppliers to consumers at
    minimum total cost, where `cost[i][j]` is the price per unit from supplier
    `i` to consumer `j` (`DISALLOWED` = no such route). Ships
    `min(sum(supply), sum(demand))` units.

    It is solved exactly by turning every unit into its own row/column of an
    assignment problem, so it is meant for modest quantities: more than
    `max_units` units raises `ValueError`.
    """
    for name, amounts in (("supply", supply), ("demand", demand)):
        if any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in amounts):
            raise ValueError(f"{name} must be non-negative integers")
    rows = _as_lists(cost)
    if len(rows) != len(supply) or any(len(r) != len(demand) for r in rows):
        raise ValueError(f"cost must be {len(supply)}x{len(demand)} (suppliers x consumers)")
    if max(sum(supply), sum(demand)) > max_units:
        raise ValueError(
            f"more than max_units={max_units} units; raise max_units or use an LP solver"
        )
    unit_rows = [i for i, s in enumerate(supply) for _ in range(s)]
    unit_cols = [j for j, d in enumerate(demand) for _ in range(d)]
    if not unit_rows or not unit_cols:
        return Transport({}, 0, 0)
    try:
        pairs = solve([[rows[i][j] for j in unit_cols] for i in unit_rows]).pairs
    except UnsolvableMatrix:
        raise UnsolvableMatrix(
            "the forbidden routes make it impossible to ship all possible units"
        ) from None
    flows: dict[tuple[int, int], int] = {}
    for r, c in pairs:
        key = (unit_rows[r], unit_cols[c])
        flows[key] = flows.get(key, 0) + 1
    total = sum(rows[i][j] * units for (i, j), units in flows.items())
    return Transport(dict(sorted(flows.items())), total, len(pairs))


# ---------------------------------------------------------------------------
# Entropic optimal transport (Sinkhorn)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SinkhornResult:
    """Result of `sinkhorn`: the transport `plan`, its cost, and convergence info."""

    plan: tuple[tuple[float, ...], ...]
    cost: float
    iterations: int
    converged: bool


def _logsumexp(terms: list[float]) -> float:
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def sinkhorn(
    cost: MatrixLike,
    *,
    reg: float = 0.1,
    a: Sequence[float] | None = None,
    b: Sequence[float] | None = None,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> SinkhornResult:
    """
    Entropy-regularised optimal transport between two distributions `a` (rows) and
    `b` (columns), solved with Sinkhorn's algorithm in the log domain (stable for
    small `reg`). Default marginals are uniform. The smaller `reg`, the closer the
    plan is to the exact (Hungarian) solution; larger `reg` spreads mass out,
    giving a *soft*, differentiable matching. `DISALLOWED`/`+inf` cells get zero
    mass. If the forbidden cells force some cell to *exactly* zero mass, the
    iteration only approaches that asymptotically and `converged` stays false.
    Pure Python: meant for small and medium problems.
    """
    n, m, edges = _validated_edges(cost)
    if n == 0 or m == 0:
        raise ValueError("cost matrix is empty")
    if not (math.isfinite(reg) and reg > 0):
        raise ValueError("reg must be a positive finite number")
    row_mass = [1.0 / n] * n if a is None else [float(x) for x in a]
    col_mass = [1.0 / m] * m if b is None else [float(x) for x in b]
    if len(row_mass) != n or len(col_mass) != m:
        raise ValueError(f"a must have {n} entries and b {m}")
    if min(row_mass) <= 0 or min(col_mass) <= 0:
        raise ValueError("a and b must be strictly positive")
    if abs(sum(row_mass) - sum(col_mass)) > 1e-9 * max(sum(row_mass), 1.0):
        raise ValueError("a and b must have the same total mass")
    by_col: list[list[tuple[int, float]]] = [[] for _ in range(m)]
    by_row: list[list[tuple[int, float]]] = []
    for i, allowed in enumerate(edges):
        by_row.append([(j, float(c)) for j, c in allowed])
        for j, c in allowed:
            by_col[j].append((i, float(c)))
    if any(not r for r in by_row) or any(not c for c in by_col):
        raise ValueError("every row and column needs at least one allowed cell")

    log_a, log_b = [math.log(x) for x in row_mass], [math.log(x) for x in col_mass]
    f, g = [0.0] * n, [0.0] * m
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):  # noqa: B007
        for i in range(n):
            f[i] = reg * (log_a[i] - _logsumexp([(g[j] - c) / reg for j, c in by_row[i]]))
        for j in range(m):
            g[j] = reg * (log_b[j] - _logsumexp([(f[i] - c) / reg for i, c in by_col[j]]))
        err = max(
            abs(sum(math.exp((f[i] + g[j] - c) / reg) for j, c in by_row[i]) - row_mass[i])
            for i in range(n)
        )
        if err < tol:
            converged = True
            break
    plan = [[0.0] * m for _ in range(n)]
    total = 0.0
    for i in range(n):
        for j, c in by_row[i]:
            plan[i][j] = math.exp((f[i] + g[j] - c) / reg)
            total += plan[i][j] * c
    return SinkhornResult(tuple(tuple(r) for r in plan), total, iterations, converged)


def soft_assignment(
    matrix: MatrixLike, temperature: float = 0.1, *, max_iter: int = 5000
) -> list[list[float]]:
    """
    A *soft* assignment: `result[i][j]` is the share of row `i` given to column
    `j`. Rows sum to 1 (for a square matrix, columns do too: a doubly stochastic
    matrix). As `temperature` -> 0 it approaches the Hungarian solution's 0/1
    matrix; larger values express uncertainty between near-equal choices. Raises
    `ValueError` if it does not converge.
    """
    n = len(_as_lists(matrix))
    result = sinkhorn(matrix, reg=temperature, max_iter=max_iter)
    if not result.converged:
        raise ValueError("soft_assignment did not converge; raise the temperature")
    return [[n * share for share in row] for row in result.plan]
