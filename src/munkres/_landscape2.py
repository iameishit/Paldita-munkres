# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Energy landscape of near-optimal assignments, diversity, temperature paths and canonical-optimum proofs (small problems)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from munkres._core import _as_lists, _is_forbidden

__all__ = [
    "cooling_schedule",
    "energy_barrier",
    "energy_landscape",
    "k_best_diverse",
    "temperature_path",
    "verify_canonical",
]


def _neighbours(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    """One move apart: one row changes to an unused column, or two rows swap columns."""
    da = dict(a)
    db = dict(b)
    diff = [r for r in da if da[r] != db.get(r)]
    if len(diff) == 1:
        return db[diff[0]] not in da.values()
    if len(diff) == 2:
        r1, r2 = diff
        return da[r1] == db[r2] and da[r2] == db[r1]
    return False


def energy_landscape(
    matrix: Any, eps: Any, maximize: bool = False, limit: int = 5000
) -> dict[str, Any]:
    """Nodes = every assignment within `eps` of optimal (with its gap); edges = single moves between them."""
    from munkres._sensitivity import near_optimal

    nodes = near_optimal(matrix, eps, maximize, limit)
    edges = [
        (i, j)
        for i in range(len(nodes))
        for j in range(i + 1, len(nodes))
        if _neighbours(nodes[i][1], nodes[j][1])
    ]
    return {"nodes": [{"gap": g, "pairs": p} for g, p in nodes], "edges": edges}


def energy_barrier(
    matrix: Any, a: Any, b: Any, eps: Any, maximize: bool = False, limit: int = 5000
) -> Any:
    """
    Smallest `h` such that assignments `a` and `b` are connected by single moves through assignments that are all
    within `h` of the optimum (the barrier); None if they are not connected within `eps`. Exact.
    """
    land = energy_landscape(matrix, eps, maximize, limit)
    idx = {n["pairs"]: k for k, n in enumerate(land["nodes"])}
    a, b = tuple(sorted(map(tuple, a))), tuple(sorted(map(tuple, b)))
    if a not in idx or b not in idx:
        raise ValueError("a and b must be assignments within eps of the optimum")
    order = sorted(range(len(land["nodes"])), key=lambda k: land["nodes"][k]["gap"])
    parent = list(range(len(order)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: dict[int, list[int]] = {}
    for i, j in land["edges"]:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    active: set[int] = set()
    for k in order:
        active.add(k)
        for nb in adj.get(k, []):
            if nb in active:
                parent[find(k)] = find(nb)
        if (
            a in idx
            and b in idx
            and idx[a] in active
            and idx[b] in active
            and find(idx[a]) == find(idx[b])
        ):
            return land["nodes"][k]["gap"]
    return None


def k_best_diverse(
    matrix: Any, k: int, min_distance: int, maximize: bool = False, limit: int = 20000
) -> list[tuple[Any, tuple[tuple[int, int], ...]]]:
    """
    Greedy diverse k-best: scan assignments by increasing (gap, pairs) and keep one if it differs from every kept
    assignment in at least `min_distance` rows. Each pick is the best assignment compatible with the earlier picks
    (greedy, not a global optimum over sets). Exact enumeration, small problems.
    """
    from munkres._landscape import matching_distance
    from munkres._sensitivity import near_optimal

    top = near_optimal(matrix, 10**9, maximize, limit)
    kept: list[tuple[Any, tuple[tuple[int, int], ...]]] = []
    for gap, pairs in top:
        if all(matching_distance(pairs, q) >= min_distance for _, q in kept):
            kept.append((gap, pairs))
            if len(kept) == k:
                break
    return kept


def cooling_schedule(start: float, end: float, steps: int) -> list[float]:
    """Geometric inverse-temperature schedule from `start` to `end` (inclusive)."""
    if steps < 2 or start <= 0 or end < start:
        raise ValueError("need steps >= 2 and 0 < start <= end")
    r = (end / start) ** (1 / (steps - 1))
    return [start * r**k for k in range(steps)]


def temperature_path(matrix: Any, betas: Any) -> list[dict[str, float]]:
    """Exact Gibbs statistics (expected cost, entropy, log Z, largest edge marginal) along a list of betas."""
    from munkres._gibbs import gibbs

    out = []
    for b in betas:
        g = gibbs(matrix, b)
        out.append(
            {
                "beta": b,
                "expected_cost": g.expected_cost,
                "entropy": g.entropy,
                "log_z": g.log_z,
                "max_marginal": max((x for r in g.marginals for x in r), default=0.0),
            }
        )
    return out


def verify_canonical(matrix: Any, pairs: Any, maximize: bool = False) -> Any:
    """
    Independent exact proof that `pairs` is the lexicographically smallest optimal assignment: it must be optimal, and for
    every row, using any smaller allowed column (with the earlier rows fixed) must be infeasible or strictly worse.
    """
    from munkres._api import solve
    from munkres._core import UnsolvableMatrix, _validated_edges
    from munkres._verify import Verification, verify_pairs

    rows = _as_lists(matrix)
    base = verify_pairs(rows, pairs, maximize)
    if not base:
        return base
    pairs = sorted(map(tuple, pairs))
    n, m, _edges = _validated_edges(rows)
    opt = base.objective
    if opt is None:
        raise ValueError("verification did not report an objective")
    sign = -1 if maximize else 1
    fixed: list[tuple[int, int]] = []
    for i, j in pairs:
        for j2 in range(j):
            if _is_forbidden(rows[i][j2]):
                continue
            used = {c for _, c in fixed} | {j2}
            if j2 in {c for _, c in fixed}:
                continue
            part = sum((Fraction(rows[r][c]) for r, c in fixed), Fraction(0)) + Fraction(
                rows[i][j2]
            )
            rest_rows = [r for r in range(n) if r > i]
            rest_cols = [c for c in range(m) if c not in used]
            if rest_rows:
                sub = [[rows[r][c] for c in rest_cols] for r in rest_rows]
                try:
                    part += Fraction(solve(sub, maximize=maximize).total) if sub and sub[0] else 0
                except UnsolvableMatrix:
                    continue
            if sign * (part - Fraction(opt)) <= 0:
                return Verification(
                    False, "canonical", f"row {i} could use smaller column {j2} at equal cost"
                )
        fixed.append((i, j))
    return Verification(True, "ok", "lexicographically smallest optimum", opt)
