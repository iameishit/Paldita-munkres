# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Planner: observable structure -> a deterministic, explainable execution strategy.

The planner only *chooses among exact algorithms*; it never relaxes correctness. Every rule
is a plain fact test (no learned model). With `verify=True` the result is independently
proven optimal by `_verify` before it is returned.

Strategies: "empty", "components" (>1 connected component), "native-kernel" (dense C kernel),
"python-exact" (pure-Python). Monge structure is *reported* but never auto-selected: measured on
this repo's benchmark (n=100..400) the exact Monge DP is 12-25x slower than the native kernel, so
`solve_monge` is an explicit, certificate-bearing tool, not a speed rule.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from munkres import _native
from munkres._core import _as_lists, _is_forbidden
from munkres._structure import Fingerprint, fingerprint

__all__ = [
    "EquivalenceReport",
    "Plan",
    "PlannedResult",
    "Transform",
    "backends",
    "check_equivalent",
    "plan",
    "profile_solve",
    "solve_planned",
    "versions",
]


@dataclass(frozen=True)
class Transform:
    """One entry of the transformation ledger (what was changed, and how it is undone)."""

    name: str
    detail: str
    reversible: str


@dataclass(frozen=True)
class Plan:
    strategy: str
    reasons: tuple[str, ...]
    fingerprint: Fingerprint
    ledger: tuple[Transform, ...] = field(default=())
    obligations: tuple[tuple[str, bool, str], ...] = field(default=())

    def explain(self) -> str:
        lines = [f"strategy: {self.strategy}"] + [f"  - {r}" for r in self.reasons]
        lines += [f"  transform {t.name}: {t.detail} (undo: {t.reversible})" for t in self.ledger]
        return "\n".join(lines)


@dataclass(frozen=True)
class PlannedResult:
    pairs: tuple[tuple[int, int], ...]
    total: Any
    plan: Plan
    certificate: Any = None


def plan(cost: Any, maximize: bool = False) -> Plan:
    """Choose an execution strategy from the matrix structure and explain why (`Plan.explain()`)."""
    fp = fingerprint(cost)
    n, m = fp.shape
    reasons: list[str] = [f"shape {n}x{m}, density {fp.density:.3f}, components {fp.components}"]
    ledger: list[Transform] = []
    if maximize:
        ledger.append(
            Transform("negate", "maximise solved as minimise of -cost", "objective = -objective")
        )
    if n > m:
        ledger.append(
            Transform(
                "orient",
                "more rows than columns: roles of rows/cols swapped internally",
                "swap pair order",
            )
        )
    obligations: list[tuple[str, bool, str]] = []
    cells = [v for r in _as_lists(cost) for v in r if not _is_forbidden(v)]
    finite = [-v for v in cells] if maximize else cells
    native_types = all(type(v) in (int, float, bool) for v in finite)
    mx = max((abs(v) for v in finite), default=0)
    scale = n + m + 4
    if not native_types:
        safe = False
    elif any(
        type(v) is float for v in finite
    ):  # float kernel: duals stay finite, ints must be exact in float64
        safe = mx * scale < 1.7e308 and all(
            abs(v) <= 1 << 53 for v in finite if type(v) is not float
        )
    else:  # int64 kernel: the dual potentials must stay inside the int64 safe range (no float conversion)
        safe = mx * scale < 1 << 62
    obligations.append(
        (
            "cost cells are int/float (native kernel domain)",
            native_types,
            "required before using the C kernel",
        )
    )
    obligations.append(("max|c|*(n+m+4) is below the overflow bound", safe, f"max|c|={mx}"))
    if n == 0 or m == 0:
        return Plan(
            "empty",
            (*reasons, "no cells: empty assignment"),
            fp,
            tuple(ledger),
            tuple(obligations),
        )
    if fp.components > 1 and (fp.components - fp.isolated_rows - fp.isolated_columns) > 1:
        reasons.append(f"{fp.components} independent components: solved separately (exact)")
        obligations.append(
            (
                "each connected component is solved independently",
                True,
                f"{fp.components} components found by union-find",
            )
        )
        return Plan("components", tuple(reasons), fp, tuple(ledger), tuple(obligations))
    if fp.monge or fp.inverse_monge:
        reasons.append(
            "proved Monge/inverse-Monge (exact); `solve_monge` is available but slower than the kernel, so not used"
        )
    if _native.available() and native_types and safe:
        reasons.append("native C kernel available and its numeric obligations hold")
        return Plan("native-kernel", tuple(reasons), fp, tuple(ledger), tuple(obligations))
    reasons.append(
        "pure-Python exact solver"
        + (" (a native obligation failed)" if _native.available() else " (no native kernel)")
    )
    return Plan("python-exact", tuple(reasons), fp, tuple(ledger), tuple(obligations))


def solve_planned(cost: Any, maximize: bool = False, verify: bool = False) -> PlannedResult:
    """Plan, execute, and (optionally) certify. The returned pairs are always an optimal assignment."""
    from munkres._api import solve
    from munkres._components import solve_by_components

    rows = _as_lists(cost)
    p = plan(rows, maximize)
    if p.strategy == "empty":
        return PlannedResult((), 0, p)
    pairs: Sequence[tuple[int, int]]
    if p.strategy == "components":
        pairs, total, _ = solve_by_components(rows, maximize)
    else:
        res = solve(rows, maximize=maximize)
        pairs, total = res.pairs, res.total
    cert = None
    if verify:
        from munkres._verify import Certificate, certify

        out = certify(rows, pairs, maximize)
        if not isinstance(out, Certificate):
            raise AssertionError(
                f"internal error: planner result failed verification: {out.reason}"
            )
        cert = out
    return PlannedResult(tuple(sorted(pairs)), total, p, cert)


def versions() -> dict[str, Any]:
    """Separate version numbers for the package, public API, solver semantics, certificate format and native ABI."""
    from munkres import __version__
    from munkres._verify import CERT_VERSION

    return {
        "package": __version__,
        "api": "3.0",
        "semantics": 1,
        "certificate": CERT_VERSION,
        "native_abi": 1,
    }


def backends() -> list[dict[str, Any]]:
    """Capability descriptors of the execution backends (what each one supports, and its guarantee)."""
    return [
        {
            "name": "native-kernel",
            "available": _native.available(),
            "dtypes": ["float64", "int64"],
            "forbidden_cells": True,
            "rectangular": True,
            "exact_arithmetic": False,
            "duals": False,
            "trace": False,
            "guarantee": "optimal for the stated float64/int64 data; int64 exact; declines on overflow risk",
        },
        {
            "name": "python-exact",
            "available": True,
            "dtypes": ["int", "float", "int64", "float64", "Fraction", "Decimal"],
            "forbidden_cells": True,
            "rectangular": True,
            "exact_arithmetic": True,
            "duals": True,
            "trace": True,
            "guarantee": "optimal; exact for int/Fraction/Decimal; float arithmetic escalates to Fraction on overflow risk",
        },
        {
            "name": "components",
            "available": True,
            "dtypes": ["int", "float", "Fraction", "Decimal"],
            "forbidden_cells": True,
            "rectangular": True,
            "exact_arithmetic": True,
            "duals": False,
            "trace": False,
            "guarantee": "same optimum as python-exact, solved per connected component",
        },
    ]


def profile_solve(cost: Any, maximize: bool = False) -> dict[str, Any]:
    """Stage timeline in nanoseconds (validate, fingerprint, plan, solve, verify) for one planned, verified solve."""
    import time

    from munkres._core import _validated_edges
    from munkres._verify import certify

    t = [time.perf_counter_ns()]
    rows = _as_lists(cost)
    _validated_edges(rows)
    t.append(time.perf_counter_ns())
    fingerprint(rows)
    t.append(time.perf_counter_ns())
    p = plan(rows, maximize)
    t.append(time.perf_counter_ns())
    res = solve_planned(rows, maximize)
    t.append(time.perf_counter_ns())
    ok = bool(certify(rows, res.pairs, maximize))
    t.append(time.perf_counter_ns())
    names = ["validate", "fingerprint", "plan", "solve", "verify"]
    return {
        "strategy": p.strategy,
        "verified": ok,
        "stages_ns": dict(zip(names, (b - a for a, b in itertools.pairwise(t)), strict=False)),
        "total_ns": t[-1] - t[0],
    }


@dataclass(frozen=True)
class EquivalenceReport:
    """Result of `check_equivalent`."""

    agree: bool
    objectives: dict[str, Any]
    verified: bool


def check_equivalent(cost: Any, maximize: bool = False) -> EquivalenceReport:
    """
    Solve the same problem with every independent route (default dispatch, pure Python, components, the planner,
    canonical optimum, tropical subset DP when small) and certify the answer; `agree` is True iff all objectives match.
    """
    from munkres._algebra import tropical_permanent
    from munkres._api import solve
    from munkres._components import solve_by_components
    from munkres._landscape import canonical_optimum
    from munkres._verify import certify

    rows = _as_lists(cost)
    n, m = len(rows), len(rows[0]) if rows else 0
    obj: dict[str, Any] = {}
    res = solve(rows, maximize=maximize)
    obj["default"] = res.total
    with _native.forced_off():
        obj["python"] = solve(rows, maximize=maximize).total
    obj["components"] = solve_by_components(rows, maximize)[1]
    obj["planner"] = solve_planned(rows, maximize).total
    if 0 < n <= m:
        obj["canonical"] = sum((rows[i][j] for i, j in canonical_optimum(rows, maximize)), 0)
        if m <= 16:
            obj["tropical"] = tropical_permanent(rows, maximize)
    vals = list(obj.values())
    agree = all(v == vals[0] for v in vals)
    return EquivalenceReport(agree, obj, bool(certify(rows, res.pairs, maximize)))
