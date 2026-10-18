# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Assignment as a market (unit-demand buyers, indivisible items) and scenario-based assignment.

`market_equilibrium` returns a welfare-maximising assignment together with *competitive equilibrium
prices* read off the LP duals: every buyer gets an item that maximises value - price among all items
(or nothing, with utility 0), and unsold items cost 0. The prices are one equilibrium, not
necessarily the minimum one. `is_competitive_equilibrium` re-checks that from scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists

__all__ = [
    "Market",
    "is_competitive_equilibrium",
    "market_equilibrium",
    "prune_dominated_scenarios",
    "robust_edges",
    "scenario_assignment",
]


def _forbidden(v: Any) -> bool:
    return v is DISALLOWED or v is None or (isinstance(v, float) and v == -float("inf"))


@dataclass(frozen=True)
class Market:
    """Result of `market_equilibrium`."""

    assignment: dict[int, int | None]  # buyer -> item (None = buys nothing)
    prices: tuple[Fraction, ...]
    utilities: tuple[Fraction, ...]
    welfare: Fraction


def market_equilibrium(valuations: Any) -> Market:
    """Welfare-maximising assignment and competitive-equilibrium prices. `DISALLOWED`/None/-inf = not interested."""
    from munkres._api import solve
    from munkres._verify import Certificate, certify

    val = _as_lists(valuations)
    n, m = len(val), len(val[0]) if val else 0
    cost = [
        [DISALLOWED if _forbidden(v) else -Fraction(v) for v in r] + [Fraction(0)] * (n + 1)
        for r in val
    ]
    res = solve(cost)
    cert = certify(cost, res.pairs)
    if not isinstance(cert, Certificate):
        raise AssertionError(cert.reason)
    assign = {i: (j if j < m else None) for i, j in res.pairs}
    prices = tuple(-cert.v[j] for j in range(m))
    util = tuple(-cert.u[i] for i in range(n))
    welfare = sum((Fraction(val[i][j]) for i, j in assign.items() if j is not None), Fraction(0))
    return Market(assign, prices, util, welfare)


def is_competitive_equilibrium(
    valuations: Any, assignment: dict[int, int | None], prices: Sequence[Any]
) -> bool:
    """Independent check: demand is satisfied, unsold items are free, nobody pays more than they value."""
    val = _as_lists(valuations)
    n, m = len(val), len(val[0]) if val else 0
    p = [Fraction(x) for x in prices]
    if any(x < 0 for x in p):
        return False
    sold = {j for j in assignment.values() if j is not None}
    if len(sold) != sum(1 for j in assignment.values() if j is not None):
        return False  # an item sold twice
    if any(p[j] != 0 for j in range(m) if j not in sold):
        return False
    for i in range(n):
        j = assignment.get(i)
        best = max(
            (Fraction(val[i][k]) - p[k] for k in range(m) if not _forbidden(val[i][k])),
            default=Fraction(0),
        )
        best = max(best, Fraction(0))
        if j is None:
            if best != 0:
                return False
        elif _forbidden(val[i][j]) or Fraction(val[i][j]) - p[j] != best:
            return False
    return True


def scenario_assignment(
    scenarios: Sequence[Any], weights: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Assignment minimising the weighted *expected* cost over scenarios (exactly one LAP on the weighted mean)."""
    from munkres._api import solve

    mats = [_as_lists(s) for s in scenarios]
    k = len(mats)
    w = [Fraction(1, k)] * k if weights is None else [Fraction(x) for x in weights]
    if len(w) != k or any(x < 0 for x in w) or sum(w) == 0:
        raise ValueError("need one non-negative weight per scenario with a positive sum")
    tot = sum(w)
    n, m = len(mats[0]), len(mats[0][0])
    mean = [
        [
            DISALLOWED
            if any(s[i][j] is DISALLOWED for s in mats)
            else sum((wk * Fraction(s[i][j]) for wk, s in zip(w, mats, strict=False)), Fraction(0))
            / tot
            for j in range(m)
        ]
        for i in range(n)
    ]
    res = solve(mean)
    pairs = tuple(sorted(res.pairs))
    return {
        "pairs": pairs,
        "expected_cost": res.total,
        "scenario_costs": [sum((s[i][j] for i, j in pairs), 0) for s in mats],
    }


def prune_dominated_scenarios(scenarios: Sequence[Any]) -> list[int]:
    """
    Indices of scenarios that can matter for a *worst-case (minimax)* objective: a scenario whose every
    cell is <= another scenario's cell can never be the maximum, so it is dropped (one of equal copies kept).
    """
    mats = [_as_lists(s) for s in scenarios]

    def le(a: list[list[Any]], b: list[list[Any]]) -> bool:
        return all(
            x is DISALLOWED or (y is not DISALLOWED and x <= y)
            for ra, rb in zip(a, b, strict=False)
            for x, y in zip(ra, rb, strict=False)
            if y is not DISALLOWED or x is DISALLOWED
        ) and not any(
            y is DISALLOWED and x is not DISALLOWED
            for ra, rb in zip(a, b, strict=False)
            for x, y in zip(ra, rb, strict=False)
        )

    keep = []
    for i, a in enumerate(mats):
        dominated = any(
            j != i and le(a, mats[j]) and (not le(mats[j], a) or j < i) for j in range(len(mats))
        )
        if not dominated:
            keep.append(i)
    return keep


def robust_edges(scenarios: Sequence[Any]) -> dict[str, frozenset[tuple[int, int]]]:
    """Cells used by every optimum of every scenario (`always`) and cells optimal in every scenario (`possibly_everywhere`)."""
    from munkres._topology import topology

    tops = [topology(s) for s in scenarios]
    always = frozenset.intersection(*[t.forced for t in tops])
    anywhere = frozenset.intersection(*[t.forced | t.optional for t in tops])
    return {"always": always, "possibly_everywhere": anywhere}
