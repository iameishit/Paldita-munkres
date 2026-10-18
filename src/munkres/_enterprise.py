# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Resource limits, structured errors, declarative problem specs, cost-source registry and operational helpers."""

from __future__ import annotations

import json
import random as _random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from munkres._core import DISALLOWED, _is_forbidden

__all__ = [
    "CostSourceRegistry",
    "Namespace",
    "ProblemTooLargeError",
    "SolverLimits",
    "choose_representation",
    "estimate_density",
    "estimate_memory",
    "latency_profile",
    "namespace_of",
    "solve_limited",
    "solve_spec",
]


class ProblemTooLargeError(ValueError):
    """Raised *before* any allocation when a problem exceeds a `SolverLimits` bound. Has `.code` and `.details`."""

    def __init__(self, message: str, code: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.code, self.details = code, details


@dataclass(frozen=True)
class SolverLimits:
    max_rows: int | None = None
    max_cols: int | None = None
    max_cells: int | None = None
    max_memory_bytes: int | None = None


def estimate_memory(n: int, m: int, representation: str = "dense", density: float = 1.0) -> int:
    """Rough bytes needed: dense = 8nm; sparse = 24 per allowed cell + 16(n+m); lazy = 16(n+m) plus the candidate graph (assumed 8 per row)."""
    if representation == "dense":
        return 8 * n * m
    if representation == "sparse":
        return int(24 * n * m * density) + 16 * (n + m)
    if representation == "lazy":
        return 16 * (n + m) + 24 * 8 * n
    raise ValueError("representation must be dense, sparse or lazy")


def choose_representation(
    n: int, m: int, density: float = 1.0, budget_bytes: int = 2 << 30
) -> dict[str, Any]:
    """Pick dense / sparse / lazy from shape, density and a memory budget, with the reasons."""
    est = {r: estimate_memory(n, m, r, density) for r in ("dense", "sparse", "lazy")}
    if est["dense"] <= budget_bytes and density > 0.3:
        rep, why = (
            "dense",
            f"dense needs {est['dense']} B <= budget and density {density:.2f} > 0.3",
        )
    elif est["sparse"] <= budget_bytes:
        rep, why = "sparse", f"sparse needs {est['sparse']} B <= budget"
    else:
        rep, why = (
            "lazy",
            "even the sparse form exceeds the budget: evaluate costs on demand (solve_lazy)",
        )
    return {"representation": rep, "memory_estimates": est, "reasons": [why]}


def solve_limited(
    cost: Any, limits: SolverLimits, maximize: bool = False, verify: bool = False
) -> Any:
    """`solve_planned` after checking shape-based limits (no allocation happens before the checks)."""
    from munkres._plan import solve_planned

    shape = getattr(cost, "shape", None)
    n, m = (
        (int(shape[0]), int(shape[1]))
        if shape is not None and len(shape) == 2
        else (len(cost), len(cost[0]) if len(cost) else 0)
    )
    det = {"rows": n, "cols": m, "cells": n * m}
    if limits.max_rows is not None and n > limits.max_rows:
        raise ProblemTooLargeError(
            f"{n} rows exceed the limit of {limits.max_rows}", "ROWS_LIMIT", det
        )
    if limits.max_cols is not None and m > limits.max_cols:
        raise ProblemTooLargeError(
            f"{m} columns exceed the limit of {limits.max_cols}", "COLS_LIMIT", det
        )
    if limits.max_cells is not None and n * m > limits.max_cells:
        raise ProblemTooLargeError(
            f"{n * m} cells exceed the limit of {limits.max_cells}", "CELLS_LIMIT", det
        )
    if limits.max_memory_bytes is not None and estimate_memory(n, m) > limits.max_memory_bytes:
        raise ProblemTooLargeError(
            f"dense form needs about {estimate_memory(n, m)} bytes",
            "MEMORY_LIMIT",
            {**det, "estimated_bytes": estimate_memory(n, m)},
        )
    return solve_planned(cost, maximize, verify)


def latency_profile(fn: Callable[[], Any], repeat: int = 200, warmup: int = 10) -> dict[str, float]:
    """Per-call latency percentiles in nanoseconds (p50, p90, p95, p99, p99.9, max) after a warm-up."""
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(repeat):
        t = time.perf_counter_ns()
        fn()
        xs.append(time.perf_counter_ns() - t)
    xs.sort()

    def pick(q: float) -> float:
        return float(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))])

    return {
        "n": float(len(xs)),
        "mean": sum(xs) / len(xs),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "p999": pick(0.999),
        "max": float(xs[-1]),
    }


def estimate_density(cost: Any, samples: int = 1000, rng: Any = None) -> tuple[float, float]:
    """Estimated fraction of allowed cells from `samples` random cells, and its standard error (selectivity estimate)."""
    rng = rng or _random.Random()  # noqa: S311  # nosec B311
    n, m = len(cost), len(cost[0])
    hits = 0
    for _ in range(samples):
        v = cost[rng.randrange(n)][rng.randrange(m)]
        hits += not (v is None or _is_forbidden(v))
    p = hits / samples
    return p, (p * (1 - p) / samples) ** 0.5


def _num(v: Any) -> Any:
    from fractions import Fraction

    return str(v) if isinstance(v, Fraction) else v


def solve_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Declarative interface. `spec` = {"cost": [[number | null]], "objective": "min" | "max", "forbidden": [[i, j], ...],
    "verify": bool, "limits": {...}}. `null` cells are forbidden. Returns a JSON-safe dict. Raises ValueError with a clear message on bad input.
    """
    from munkres._plan import solve_planned
    from munkres._verify import problem_hash

    if not isinstance(spec, dict) or "cost" not in spec:
        raise ValueError("spec must be a dict with a 'cost' key")
    unknown = set(spec) - {"cost", "objective", "forbidden", "verify", "limits"}
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(map(str, unknown))}")
    if "verify" in spec and type(spec["verify"]) is not bool:
        raise ValueError("'verify' must be true or false")
    forb = spec.get("forbidden", [])
    if not isinstance(forb, list) or any(
        not (isinstance(c, list) and len(c) == 2 and all(type(x) is int for x in c)) for c in forb
    ):
        raise ValueError("'forbidden' must be a list of [row, col] integer pairs")
    lims = spec.get("limits")
    if lims is not None and not (
        isinstance(lims, dict)
        and set(lims) <= {"max_rows", "max_cols", "max_cells", "max_memory_bytes"}
        and all(v is None or (type(v) is int and v >= 0) for v in lims.values())
    ):
        raise ValueError(
            "'limits' must be a dict of non-negative integers (max_rows, max_cols, max_cells, max_memory_bytes)"
        )
    cost = spec["cost"]
    if (
        not isinstance(cost, list)
        or not cost
        or any(not isinstance(r, list) or len(r) != len(cost[0]) for r in cost)
    ):
        raise ValueError("'cost' must be a non-empty rectangular list of lists")
    obj = spec.get("objective", "min")
    if obj not in ("min", "max"):
        raise ValueError("'objective' must be 'min' or 'max'")
    rows = [[DISALLOWED if v is None else v for v in r] for r in cost]
    for cell in spec.get("forbidden", []):
        i, j = cell
        if not (0 <= i < len(rows) and 0 <= j < len(rows[0])):
            raise ValueError(f"forbidden cell {cell} is out of range")
        rows[i][j] = DISALLOWED
    for r in rows:
        for v in r:
            if v is not DISALLOWED and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ValueError(f"cost cells must be numbers or null, got {v!r}")
            if isinstance(v, float) and v != v:  # noqa: PLR0124
                raise ValueError("cost cells must not be NaN")
    lim = SolverLimits(**spec["limits"]) if spec.get("limits") else None
    maximize = obj == "max"
    res = (
        solve_limited(rows, lim, maximize, bool(spec.get("verify")))
        if lim
        else solve_planned(rows, maximize, bool(spec.get("verify")))
    )
    out = {
        "pairs": [list(p) for p in res.pairs],
        "total": _num(res.total),
        "strategy": res.plan.strategy,
        "reasons": list(res.plan.reasons),
        "problem_hash": problem_hash(rows, maximize),
    }
    if res.certificate is not None:
        out["certificate"] = json.loads(res.certificate.to_json())
    return out


class Namespace:
    """Read-only virtual tree over a solved problem (`/problem`, `/solution`, `/plan`, `/certificate`, `/backends`)."""

    def __init__(self, tree: dict[str, Any]) -> None:
        self._t = tree

    def _walk(self, path: str) -> Any:
        node: Any = self._t
        for part in [p for p in path.split("/") if p]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(path)
            node = node[part]
        return node

    def ls(self, path: str = "/") -> list[str]:
        node = self._walk(path)
        if not isinstance(node, dict):
            raise NotADirectoryError(path)
        return sorted(node)

    def read(self, path: str) -> Any:
        node = self._walk(path)
        if isinstance(node, dict):
            raise IsADirectoryError(path)
        return node


def namespace_of(matrix: Any, maximize: bool = False) -> Namespace:
    """Solve `matrix` with verification and expose everything as a `Namespace`."""
    from munkres._plan import backends, solve_planned
    from munkres._verify import problem_hash

    res = solve_planned(matrix, maximize, verify=True)
    n, m = res.plan.fingerprint.shape
    return Namespace(
        {
            "problem": {
                "shape": (n, m),
                "hash": problem_hash(matrix, maximize),
                "maximize": maximize,
            },
            "solution": {"pairs": list(res.pairs), "total": res.total},
            "plan": {"strategy": res.plan.strategy, "reasons": list(res.plan.reasons)},
            "certificate": {
                "json": res.certificate.to_json(),
                "objective": str(res.certificate.objective),
            },
            "backends": {
                b["name"]: {k: str(v) for k, v in b.items() if k != "name"} for b in backends()
            },
        }
    )


class CostSourceRegistry:
    """Mount named cost sources (matrix, ndarray, or `cost(i, j)` callable with a shape) and solve them by name."""

    def __init__(self) -> None:
        self._s: dict[str, tuple[Any, tuple[int, int] | None]] = {}

    def mount(self, name: str, source: Any, shape: tuple[int, int] | None = None) -> None:
        if callable(source) and shape is None:
            raise ValueError("a callable source needs a shape")
        self._s[name] = (source, shape)

    def names(self) -> list[str]:
        return sorted(self._s)

    def solve(self, name: str, maximize: bool = False) -> Any:
        from munkres._lazy import solve_lazy
        from munkres._plan import solve_planned

        src, shape = self._s[name]
        if callable(src):
            if maximize:
                raise ValueError(
                    "maximize is not supported for callable sources; negate the cost function"
                )
            if shape is None:
                raise ValueError("callable cost sources need an explicit (rows, cols) shape")
            return solve_lazy(shape[0], shape[1], src)
        return solve_planned(src, maximize)
