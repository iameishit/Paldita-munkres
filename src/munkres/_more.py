# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Decision-support tools, exact QAP branch and bound, parametric extras, history-based warm starts, grid / out-of-core solving."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, UnsolvableMatrix, _as_lists, _is_forbidden

__all__ = [
    "HistoryCache",
    "ParametricCache",
    "calibrate_native_threshold",
    "classify_cost",
    "compact_log",
    "fit_threshold",
    "fragility_graph",
    "hall_violators",
    "next_breakpoint",
    "optimality_length",
    "parametric_atlas",
    "parametric_degeneracy",
    "parametric_duals",
    "pareto_assignments",
    "qap_branch_and_bound",
    "qap_gap",
    "scenario_mass",
    "solve_group_capacity",
    "solve_memmap",
    "solve_points_grid",
    "stable_optimum",
    "strip_additive",
    "var_assignment",
]


def fragility_graph(matrix: Any, maximize: bool = False) -> list[dict[str, Any]]:
    """For each cell of the optimum: the best alternative assignment if that cell were forbidden, and the exact penalty (None if infeasible)."""
    from munkres._api import solve

    rows = _as_lists(matrix)
    base = solve(rows, maximize=maximize)
    out = []
    for i, j in sorted(base.pairs):
        alt = [list(r) for r in rows]
        alt[i][j] = DISALLOWED
        try:
            a = solve(alt, maximize=maximize)
            out.append(
                {
                    "cell": (i, j),
                    "alternative": tuple(sorted(a.pairs)),
                    "penalty": abs(Fraction(a.total) - Fraction(base.total)),
                }
            )
        except UnsolvableMatrix:
            out.append({"cell": (i, j), "alternative": None, "penalty": None})
    return out


def scenario_mass(scenarios: Sequence[Any]) -> dict[tuple[tuple[int, int], ...], Fraction]:
    """For every assignment that is optimal in some scenario: the fraction of scenarios in which it is optimal."""
    from munkres._landscape import enumerate_optima

    mass: dict[Any, Fraction] = {}
    for s in scenarios:
        for sol in enumerate_optima(s):
            mass[sol] = mass.get(sol, Fraction(0)) + Fraction(1, len(scenarios))
    return mass


def _scen(scenarios: Sequence[Any], max_n: int = 8) -> tuple[list[Any], int, int]:
    mats = [_as_lists(s) for s in scenarios]
    n, m = len(mats[0]), len(mats[0][0])
    if n > m or n > max_n:
        raise ValueError(f"supports rows <= cols and n <= {max_n} (enumeration)")
    return mats, n, m


def pareto_assignments(scenarios: Sequence[Any]) -> list[tuple[tuple[int, int], ...]]:
    """Assignments whose scenario-cost vector is not dominated (no other is <= in every scenario and < in one). Exact enumeration, n <= 8."""
    mats, n, m = _scen(scenarios)
    vecs = {}
    for cols in itertools.permutations(range(m), n):
        if all(s[i][c] is not DISALLOWED for s in mats for i, c in enumerate(cols)):
            vecs[cols] = tuple(sum(s[i][c] for i, c in enumerate(cols)) for s in mats)
    out = []
    for p, v in vecs.items():
        if not any(
            w != v and all(x <= y for x, y in zip(w, v, strict=False)) for w in vecs.values()
        ):
            out.append(tuple(enumerate(p)))
    return sorted(out)


def var_assignment(
    scenarios: Sequence[Any], delta: float
) -> tuple[tuple[tuple[int, int], ...], Any]:
    """Minimise Value-at-Risk: the smallest B with P(cost <= B) >= 1 - delta over equally likely scenarios (the ceil((1-delta) S)-th smallest cost)."""
    mats, n, m = _scen(scenarios)
    k = max(1, math.ceil((1 - delta) * len(mats) - 1e-12))
    best = None
    for cols in itertools.permutations(range(m), n):
        if all(s[i][c] is not DISALLOWED for s in mats for i, c in enumerate(cols)):
            v = sorted(sum(s[i][c] for i, c in enumerate(cols)) for s in mats)[k - 1]
            if best is None or v < best[1]:
                best = (tuple(enumerate(cols)), v)
    if best is None:
        raise UnsolvableMatrix("no feasible assignment")
    return best


def qap_gap(flow: Any, dist: Any, restarts: int = 5, seed: int = 0) -> dict[str, Any]:
    """Gilmore-Lawler bound vs the best 2-swap local-search incumbent: how tight the LAP relaxation is on this instance."""
    import random

    from munkres._bounds import qap_local_search, qap_lower_bound

    n = len(flow)
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    inc = min(
        (qap_local_search(flow, dist, rng.sample(range(n), n)) for _ in range(restarts)),
        key=lambda t: t[1],
    )
    lb = qap_lower_bound(flow, dist)
    return {
        "bound": lb,
        "incumbent": inc[1],
        "gap": inc[1] - lb,
        "relative_gap": (Fraction(inc[1]) - lb) / inc[1] if inc[1] else Fraction(0),
    }


def qap_branch_and_bound(flow: Any, dist: Any) -> dict[str, Any]:
    """Exact Koopmans-Beckmann QAP by branch and bound with a Gilmore-Lawler (LAP) bound at every node. Small n (<= ~9)."""
    from munkres._api import solve
    from munkres._bounds import qap_cost, qap_local_search

    F, D = _as_lists(flow), _as_lists(dist)
    n = len(F)
    best_perm, best = qap_local_search(F, D, list(range(n)))
    nodes = 0

    def bound(p: list[int]) -> Any:
        d = len(p)
        fixed = sum(F[i][j] * D[p[i]][p[j]] for i in range(d) for j in range(d))
        free_f = list(range(d, n))
        free_l = [k for k in range(n) if k not in p]
        if not free_f:
            return fixed
        L = []
        for i in free_f:
            fi = sorted(F[i][j] for j in free_f if j != i)
            row = []
            for k in free_l:
                dk = sorted((D[k][ell] for ell in free_l if ell != k), reverse=True)
                cross = sum(F[i][j] * D[k][p[j]] + F[j][i] * D[p[j]][k] for j in range(d))
                row.append(
                    F[i][i] * D[k][k] + cross + sum(a * b for a, b in zip(fi, dk, strict=False))
                )
            L.append(row)
        return fixed + solve(L).total

    def rec(p: list[int]) -> None:
        nonlocal best, best_perm, nodes
        nodes += 1
        if len(p) == n:
            c = qap_cost(F, D, p)
            if c < best:
                best, best_perm = c, tuple(p)
            return
        if bound(p) >= best and len(p) > 0:
            return
        for k in range(n):
            if k not in p:
                rec([*p, k])

    rec([])
    return {
        "permutation": tuple(best_perm),
        "cost": best,
        "nodes": nodes,
        "brute_force_nodes": sum(math.perm(n, k) for k in range(n + 1)),
    }


def optimality_length(a: Any, b: Any, pairs: Any, lo: Any, hi: Any) -> Fraction:
    """Length of the lambda-interval on which `pairs` is optimal for A + lambda B (0 if never, or a single point)."""
    from munkres._parametric import optimality_interval

    iv = optimality_interval(a, b, pairs, lo, hi)
    return Fraction(0) if iv is None else iv[1] - iv[0]


def next_breakpoint(a: Any, b: Any, lam: Any, hi: Any) -> Fraction | None:
    """First lambda' > lam where the optimal assignment of A + lambda B changes (None if it never changes before `hi`)."""
    from munkres._parametric import parametric_assignment

    pieces = parametric_assignment(a, b, lam, hi)
    return None if pieces[0].end >= Fraction(hi) else pieces[0].end


def parametric_duals(a: Any, b: Any, lo: Any, hi: Any) -> list[dict[str, Any]]:
    """Exact optimal duals at the midpoint of every piece of the parametric optimum (checkable with `Certificate.verify`)."""
    from munkres._parametric import parametric_assignment
    from munkres._verify import Certificate, certify

    A, B = _as_lists(a), _as_lists(b)
    out = []
    for p in parametric_assignment(a, b, lo, hi):
        mid = (p.start + p.end) / 2
        mat = [
            [
                DISALLOWED if A[i][j] is DISALLOWED else Fraction(A[i][j]) + mid * Fraction(B[i][j])
                for j in range(len(A[0]))
            ]
            for i in range(len(A))
        ]
        cert = certify(mat, p.pairs)
        if not isinstance(cert, Certificate):
            raise AssertionError(cert.reason)
        out.append({"piece": p, "lambda": mid, "u": cert.u, "v": cert.v, "matrix": mat})
    return out


def parametric_degeneracy(a: Any, b: Any, lo: Any, hi: Any) -> list[tuple[Fraction, int | None]]:
    """At every breakpoint: the exact number of optimal assignments there (always >= 2, since two lines meet)."""
    from munkres._parametric import parametric_assignment
    from munkres._topology import topology

    A, B = _as_lists(a), _as_lists(b)
    out = []
    for p in parametric_assignment(a, b, lo, hi)[:-1]:
        lam = p.end
        mat = [
            [
                DISALLOWED if A[i][j] is DISALLOWED else Fraction(A[i][j]) + lam * Fraction(B[i][j])
                for j in range(len(A[0]))
            ]
            for i in range(len(A))
        ]
        out.append((lam, topology(mat).n_optimal))
    return out


def compact_log(log: dict[str, Any]) -> dict[str, Any]:
    """Drop cell updates overwritten by a later update of the same cell when no structural change happened in between."""
    ops = list(log["ops"])
    keep = [True] * len(ops)
    last: dict[tuple[Any, Any], int] = {}
    for k, op in enumerate(ops):
        if op[0] == "update_cost":
            cell = (op[1], op[2])
            if cell in last:
                keep[last[cell]] = False
            last[cell] = k
        else:
            last.clear()  # a structural op: earlier updates can no longer be merged across it
    return {**log, "ops": [op for op, k in zip(ops, keep, strict=False) if k]}


class HistoryCache:
    """Remember solved problems; solve a new same-shape problem by repairing the closest remembered state instead of starting over."""

    def __init__(self) -> None:
        self._h: list[tuple[list[list[Any]], Any]] = []

    def solve(self, matrix: Any) -> dict[str, Any]:
        from munkres._dynamic import AssignmentState

        rows = [list(r) for r in _as_lists(matrix)]

        def diff(old: list[list[Any]]) -> list[tuple[int, int]]:
            return [
                (i, j)
                for i in range(len(rows))
                for j in range(len(rows[0]))
                if old[i][j] != rows[i][j]
            ]

        cands = [
            (len(diff(o)), k)
            for k, (o, _) in enumerate(self._h)
            if len(o) == len(rows) and len(o[0]) == len(rows[0])
        ]
        if cands:
            _, k = min(cands)
            old, st0 = self._h[k]
            st = st0.snapshot()
            changed = diff(old)
            st.update_costs({c: rows[c[0]][c[1]] for c in changed})
            reused = True
        else:
            st, changed, reused = AssignmentState(rows), [], False
        st.solve()
        self._h.append((rows, st.snapshot()))
        return {
            "state": st,
            "reused": reused,
            "changed_cells": len(changed),
            "reaugmented_rows": st.stats.reaugmented_rows,
            "objective": st.objective,
        }


def solve_points_grid(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], cell: float, k: int = 4
) -> Any:
    """Exact point matching; candidates come from a spatial grid (neighbouring cells, widened until >= k) instead of sorting all targets."""
    from munkres._lazy import solve_lazy

    dim = len(x[0])
    grid: dict[tuple[int, ...], list[int]] = {}
    for j, p in enumerate(y):
        grid.setdefault(tuple(math.floor(c / cell) for c in p), []).append(j)

    def cands(i: int) -> list[int]:
        base = tuple(math.floor(c / cell) for c in x[i])
        r = 1
        while True:
            found = [
                j
                for off in itertools.product(range(-r, r + 1), repeat=dim)
                for j in grid.get(tuple(b + o for b, o in zip(base, off, strict=False)), [])
            ]
            if len(found) >= min(k, len(y)) or r > 1000:
                return found or list(range(len(y)))
            r += 1

    def d(p: Sequence[float], q: Sequence[float]) -> float:
        return math.dist(p, q) ** 2

    return solve_lazy(len(x), len(y), lambda i, j: d(x[i], y[j]), cands)


def solve_memmap(path: str, shape: tuple[int, int], k: int = 4, dtype: str = "float64") -> Any:
    """Exact assignment for a matrix stored on disk: rows are streamed from a memory map; only the candidate graph is held in memory."""
    import numpy as np

    from munkres._lazy import solve_lazy

    mm = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    n, m = shape
    return solve_lazy(
        n,
        m,
        lambda i, j: float(mm[i, j]),
        lambda i: np.argsort(mm[i], kind="stable")[: min(k, m)].tolist(),
    )


def stable_optimum(
    matrix: Any, maximize: bool = False, limit: int = 100_000
) -> tuple[tuple[tuple[int, int], ...], Any]:
    """
    Among all optimal assignments, the one whose most fragile cell is the least fragile: maximise the minimum exact exclusion penalty
    over its cells (a forced cell has infinite margin). Ties -> lexicographically smallest. Returns `(pairs, margin)`.
    """
    from munkres._landscape import enumerate_optima
    from munkres._topology import edge_margins

    margins = edge_margins(matrix, maximize)
    inf = float("inf")
    best = None
    for sol in enumerate_optima(matrix, maximize, limit):
        vals = [
            inf if margins[c]["exclude_penalty"] is None else margins[c]["exclude_penalty"]
            for c in sol
        ]
        mm = min(vals)
        if best is None or mm > best[1]:
            best = (sol, mm)
    return best  # type: ignore[return-value]


def hall_violators(matrix: Any) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every inclusion-minimal Hall violator (rows S with |N(S)| < |S|) with its neighbourhood, for rows <= cols and at most 14 rows."""
    from munkres._core import _validated_edges

    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("expects rows <= cols; pass the transpose")
    if n > 14:
        raise ValueError("enumeration limited to 14 rows")
    nb = [frozenset(j for j, _ in e) for e in edges]
    bad: list[tuple[int, ...]] = []
    for k in range(1, n + 1):
        for s in itertools.combinations(range(n), k):
            if any(set(b) <= set(s) for b in bad):
                continue  # contains a smaller violator: not minimal
            if len(frozenset().union(*(nb[i] for i in s))) < k:
                bad.append(s)
    return [(s, tuple(sorted(frozenset().union(*(nb[i] for i in s))))) for s in bad]


def classify_cost(matrix: Any) -> frozenset[str]:
    """Descriptive type tags derived from exact facts: dense/sparse, integer/float/rational, square, symmetric, monge, additive, toeplitz, ..."""
    from munkres._structure import fingerprint
    from munkres._structure2 import is_circulant, is_toeplitz

    rows = _as_lists(matrix)
    fp = fingerprint(rows)
    n, m = fp.shape
    vals = [v for r in rows for v in r if not _is_forbidden(v)]
    tags = {"dense" if fp.density > 0.5 else "sparse", "square" if n == m else "rectangular"}
    if fp.allowed_cells < n * m:
        tags.add("has_forbidden")
    if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        tags.add("integer")
    elif vals and any(isinstance(v, float) for v in vals):
        tags.add("float")
    elif vals and any(isinstance(v, Fraction) for v in vals):
        tags.add("rational")
    if vals and all(v >= 0 for v in vals):
        tags.add("nonnegative")
    for flag, tag in (
        (fp.monge, "monge"),
        (fp.inverse_monge, "inverse_monge"),
        (fp.additive, "additive"),
    ):
        if flag:
            tags.add(tag)
    if n == m and all(rows[i][j] == rows[j][i] for i in range(n) for j in range(n)):
        tags.add("symmetric")
    if n and m and is_toeplitz(rows):
        tags.add("toeplitz")
    if n and n == m and is_circulant(rows):
        tags.add("circulant")
    return frozenset(tags)


def solve_group_capacity(
    cost: Any, groups: Sequence[Any], caps: dict[Any, int], maximize: bool = False
) -> tuple[tuple[tuple[int, int], ...], Any]:
    """
    Assignment of all rows (rows <= cols) where at most `caps[g]` columns of group g may be used (a partition-matroid
    constraint). Exact depth-first search with bounding by the unconstrained optimum; intended for small instances (<= 9 rows).
    """
    rows = _as_lists(cost)
    n, m = len(rows), len(rows[0]) if rows else 0
    if n > m or n > 9 or len(groups) != m:
        raise ValueError("supports rows <= cols, at most 9 rows, one group id per column")
    sign = -1 if maximize else 1
    best: list[Any] = [None, None]

    def rec(
        i: int, used: frozenset[int], count: dict[Any, int], acc: Any, pairs: list[tuple[int, int]]
    ) -> None:
        if i == n:
            if best[0] is None or sign * acc < sign * best[0]:
                best[0], best[1] = acc, tuple(pairs)
            return
        for j in range(m):
            g = groups[j]
            if j in used or rows[i][j] is DISALLOWED or count.get(g, 0) >= caps.get(g, m):
                continue
            count[g] = count.get(g, 0) + 1
            pairs.append((i, j))
            rec(i + 1, used | {j}, count, acc + rows[i][j], pairs)
            pairs.pop()
            count[g] -= 1

    rec(0, frozenset(), {}, 0, [])
    if best[0] is None:
        raise UnsolvableMatrix("no assignment satisfies the group capacities")
    return best[1], best[0]


def parametric_atlas(
    a: Any, b1: Any, b2: Any, l1: Sequence[Any], l2: Sequence[Any], maximize: bool = False
) -> dict[str, Any]:
    """
    Optimal assignment of A + x B1 + y B2 on the grid x in `l1`, y in `l2` (exact solves): region ids per grid point, the
    assignment of each region and the exact optimum. A *sampled* phase diagram; ties on region boundaries get one arbitrary label.
    """
    from munkres._api import solve

    A, B1, B2 = _as_lists(a), _as_lists(b1), _as_lists(b2)
    regions: dict[tuple[tuple[int, int], ...], int] = {}
    labels, objective = [], []
    for x in l1:
        lrow, orow = [], []
        for y in l2:
            mat = [
                [
                    DISALLOWED
                    if A[i][j] is DISALLOWED
                    else Fraction(A[i][j])
                    + Fraction(x) * Fraction(B1[i][j])
                    + Fraction(y) * Fraction(B2[i][j])
                    for j in range(len(A[0]))
                ]
                for i in range(len(A))
            ]
            res = solve(mat, maximize=maximize)
            key = tuple(sorted(res.pairs))
            lrow.append(regions.setdefault(key, len(regions)))
            orow.append(Fraction(res.total))
        labels.append(lrow)
        objective.append(orow)
    return {"labels": labels, "regions": {v: k for k, v in regions.items()}, "objective": objective}


class ParametricCache:
    """Memoise `parametric_assignment`; a query whose interval lies inside a cached one is answered by exactly slicing the cached pieces."""

    def __init__(self) -> None:
        self._c: dict[tuple[str, str, bool], list[tuple[Fraction, Fraction, list[Any]]]] = {}
        self.hits = self.misses = 0

    def get(self, a: Any, b: Any, lo: Any, hi: Any, maximize: bool = False) -> list[Any]:
        from munkres._parametric import Piece, parametric_assignment
        from munkres._verify import problem_hash

        key = (problem_hash(a), problem_hash(b), maximize)
        lo, hi = Fraction(lo), Fraction(hi)
        for clo, chi, pieces in self._c.get(key, []):
            if clo <= lo and hi <= chi:
                self.hits += 1
                out = [
                    Piece(max(p.start, lo), min(p.end, hi), p.pairs, p.base, p.slope)
                    for p in pieces
                    if p.end >= lo and p.start <= hi
                ]
                inner = [p for p in out if p.end > p.start]
                return inner or out[:1]
        self.misses += 1
        pieces = parametric_assignment(a, b, lo, hi, maximize)
        self._c.setdefault(key, []).append((lo, hi, pieces))
        return pieces


def strip_additive(matrix: Any) -> dict[str, Any]:
    """
    Square matrices: write c[i][j] = a[i] + b[j] + h[i][j] with h's first row and column zero (exact). Because every row and
    column is used, opt(c) = sum(a) + sum(b) + opt(h) and the optimal assignments of c and h coincide. `residual` is h.
    """
    c = [[Fraction(v) for v in r] for r in _as_lists(matrix)]
    n = len(c)
    if any(len(r) != n for r in c):
        raise ValueError("square fully finite matrix required")
    a = [c[i][0] - c[0][0] for i in range(n)]
    b = [c[0][j] for j in range(n)]
    h = [[c[i][j] - a[i] - b[j] for j in range(n)] for i in range(n)]
    return {
        "row_terms": a,
        "col_terms": b,
        "residual": h,
        "constant": sum(a) + sum(b),
        "is_additive": all(v == 0 for r in h for v in r),
    }


def fit_threshold(
    samples: Sequence[tuple[float, str]], low: str = "python", high: str = "native"
) -> float:
    """Decision stump: the size threshold t minimising misclassifications of 'fastest backend' labels (`low` below t, `high` at or above)."""
    pts = sorted(samples)
    cands = (
        [pts[0][0]] + [(x + y) / 2 for (x, _), (y, _) in itertools.pairwise(pts)] + [pts[-1][0] + 1]
    )
    best = min(
        cands, key=lambda t: (sum(1 for s, lab in pts if (high if s >= t else low) != lab), t)
    )
    return float(best)


def calibrate_native_threshold(
    sizes: Sequence[int] = (2, 3, 4, 6, 8, 12, 16, 24), repeat: int = 30
) -> dict[str, Any]:
    """
    Time the pure-Python and native solvers on this machine for each matrix side n, label the faster one and learn the crossover
    cell count with `fit_threshold`. Measurement-driven (hardware-specific); it only chooses between exact solvers.
    """
    import random
    import time

    from munkres import _native
    from munkres._api import solve

    if not _native.available():
        return {"threshold_cells": None, "samples": [], "note": "native kernel not available"}
    rng = random.Random(0)  # noqa: S311  # nosec B311
    samples = []
    for n in sizes:
        a = [[rng.randint(0, 99) for _ in range(n)] for _ in range(n)]
        solve(a)
        t = time.perf_counter()
        for _ in range(repeat):
            solve(a)
        t_native = (time.perf_counter() - t) / repeat
        with _native.forced_off():
            t = time.perf_counter()
            for _ in range(repeat):
                solve(a)
            t_py = (time.perf_counter() - t) / repeat
        samples.append((n * n, "native" if t_native < t_py else "python", t_native, t_py))
    thr = fit_threshold([(s, lab) for s, lab, _, _ in samples])
    return {
        "threshold_cells": thr,
        "samples": samples,
        "note": "dispatch currently uses a fixed threshold; this is the measured crossover",
    }
