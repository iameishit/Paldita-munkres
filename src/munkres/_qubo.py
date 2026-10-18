# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
QUBO / Ising encodings of assignment, a *classical* simulated-annealing sampler and an experiment recorder.

No quantum hardware is used or claimed. The encoders are exact (verified against brute force); the sampler is a
heuristic that produces candidates, which are repaired to feasibility and checked against the exact classical optimum.
"""

from __future__ import annotations

import json
import math
import random as _random
from dataclasses import dataclass, field
from typing import Any

from munkres._core import _as_lists, _validated_edges

__all__ = [
    "QUBO",
    "ExperimentRecorder",
    "anneal_qubo",
    "assignment_to_qubo",
    "decode_bits",
    "qubo_energy",
    "qubo_to_ising",
    "reduce_assignment_problem",
    "repair_bits",
    "verify_candidate",
    "warm_start_bits",
]


@dataclass(frozen=True)
class QUBO:
    """E(x) = offset + sum_{(a, b)} Q[a, b] x_a x_b over binary x (a <= b); variable k is cell `cells[k]`."""

    Q: dict[tuple[int, int], Any]
    offset: Any
    cells: tuple[tuple[int, int], ...]
    penalty: Any
    shape: tuple[int, int]


def assignment_to_qubo(cost: Any, penalty: Any = None) -> QUBO:
    """
    Encode min-cost assignment of all rows (rows <= cols) into distinct columns. Rows: P (1 - sum_j x_ij)^2; columns:
    P x_ij x_i'j for every pair in a column. The default penalty 2 n max|c| + 1 guarantees every QUBO minimiser is a valid optimal assignment.
    """
    n, m, edges = _validated_edges(_as_lists(cost))
    if n > m:
        raise ValueError("requires rows <= cols")
    cells = tuple((i, j) for i, e in enumerate(edges) for j, _ in e)
    idx = {c: k for k, c in enumerate(cells)}
    cmax = max((abs(c) for e in edges for _, c in e), default=0)
    P = penalty if penalty is not None else 2 * n * cmax + 1
    Q: dict[tuple[int, int], Any] = {}

    def add(a: int, b: int, v: Any) -> None:
        key = (a, b) if a <= b else (b, a)
        Q[key] = Q.get(key, 0) + v

    for i, e in enumerate(edges):
        for j, c in e:
            add(idx[(i, j)], idx[(i, j)], c - P)
        for (j1, _), (j2, _) in ((p, q) for p in e for q in e if p[0] < q[0]):
            add(idx[(i, j1)], idx[(i, j2)], 2 * P)
    for j in range(m):
        rows = [i for i in range(n) if (i, j) in idx]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                add(idx[(rows[a], j)], idx[(rows[b], j)], P)
    return QUBO(Q, n * P, cells, P, (n, m))


def qubo_energy(q: QUBO, bits: Any) -> Any:
    """Energy E(x) of a bit vector under the QUBO (offset plus the sum of Q[a, b] x_a x_b)."""
    return q.offset + sum(v * bits[a] * bits[b] for (a, b), v in q.Q.items())


def decode_bits(q: QUBO, bits: Any) -> tuple[tuple[int, int], ...] | None:
    """Selected cells if they form a valid assignment (every row once, columns at most once), else None."""
    chosen = [q.cells[k] for k, b in enumerate(bits) if b]
    rows = [i for i, _ in chosen]
    cols = [j for _, j in chosen]
    if len(set(rows)) == len(rows) == q.shape[0] and len(set(cols)) == len(cols):
        return tuple(sorted(chosen))
    return None


def qubo_to_ising(q: QUBO) -> dict[str, Any]:
    """x = (1 + s) / 2: returns fields h, couplings J (a < b) and constant with E_qubo(x) == const + sum h s + sum J s s."""
    nvar = len(q.cells)
    h = [0.0] * nvar
    J: dict[tuple[int, int], float] = {}
    const = float(q.offset)
    for (a, b), v in q.Q.items():
        if a == b:
            h[a] += v / 2
            const += v / 2
        else:
            J[(a, b)] = J.get((a, b), 0.0) + v / 4
            h[a] += v / 4
            h[b] += v / 4
            const += v / 4
    return {"h": h, "J": J, "const": const}


def reduce_assignment_problem(cost: Any) -> dict[str, Any]:
    """Use the classical optimum's forced cells to shrink the problem before encoding: delete their rows/columns."""
    from munkres._topology import topology

    rows = _as_lists(cost)
    top = topology(rows)
    forced = sorted(top.forced)
    keep_r = [i for i in range(len(rows)) if i not in {r for r, _ in forced}]
    keep_c = [j for j in range(len(rows[0])) if j not in {c for _, c in forced}]
    sub = [[rows[i][j] for j in keep_c] for i in keep_r]
    return {
        "forced": forced,
        "rows": keep_r,
        "cols": keep_c,
        "matrix": sub,
        "forced_cost": sum((rows[i][j] for i, j in forced), 0),
    }


def warm_start_bits(cost: Any, q: QUBO | None = None) -> list[int]:
    """The exact classical optimum as a bit vector over the QUBO variables (a warm start for any sampler)."""
    from munkres._api import solve

    rows = _as_lists(cost)
    q = q or assignment_to_qubo(rows)
    chosen = set(solve(rows).pairs)
    return [1 if c in chosen else 0 for c in q.cells]


def repair_bits(cost: Any, q: QUBO, bits: Any) -> tuple[tuple[int, int], ...]:
    """Keep the cheapest conflict-free selected cells, then match the remaining rows optimally on the remaining columns."""
    from munkres._api import solve

    rows = _as_lists(cost)
    n, m = q.shape
    chosen = sorted(
        (c for k, c in enumerate(q.cells) if bits[k]), key=lambda c: (rows[c[0]][c[1]], c)
    )
    used_r: set[int] = set()
    used_c: set[int] = set()
    keep = []
    for i, j in chosen:
        if i not in used_r and j not in used_c:
            keep.append((i, j))
            used_r.add(i)
            used_c.add(j)
    free_r = [i for i in range(n) if i not in used_r]
    free_c = [j for j in range(m) if j not in used_c]
    if free_r:
        sub = [[rows[i][j] for j in free_c] for i in free_r]
        keep += [(free_r[a], free_c[b]) for a, b in solve(sub).pairs]
    return tuple(sorted(keep))


def anneal_qubo(
    q: QUBO, seed: int = 0, sweeps: int = 300, t0: float | None = None, t1: float = 0.05
) -> tuple[list[int], float]:
    """Single-flip Metropolis simulated annealing (deterministic for a given seed). A heuristic: no optimality claim."""
    rng = _random.Random(seed)  # noqa: S311  # nosec B311
    nvar = len(q.cells)
    nb: list[list[tuple[int, float]]] = [[] for _ in range(nvar)]
    diag = [0.0] * nvar
    for (a, b), v in q.Q.items():
        if a == b:
            diag[a] += float(v)
        else:
            nb[a].append((b, float(v)))
            nb[b].append((a, float(v)))
    x = [rng.randint(0, 1) for _ in range(nvar)]
    t0 = t0 if t0 is not None else max(1.0, float(q.penalty))
    best, best_e = x[:], float(qubo_energy(q, x))
    e = best_e
    for s in range(sweeps):
        t = t0 * (t1 / t0) ** (s / max(1, sweeps - 1))
        for _ in range(nvar):
            a = rng.randrange(nvar)
            field = diag[a] + sum(v * x[b] for b, v in nb[a])
            de = field if x[a] == 0 else -field
            if de <= 0 or rng.random() < math.exp(-de / t):
                x[a] ^= 1
                e += de
                if e < best_e - 1e-12:
                    best, best_e = x[:], e
    return best, best_e


def verify_candidate(cost: Any, pairs: Any) -> dict[str, Any]:
    """Check any candidate assignment (from any sampler) against the exact optimum: feasibility, objective, gap, provably optimal?"""
    from munkres._api import solve
    from munkres._verify import verify_pairs

    rows = _as_lists(cost)
    pairs = tuple(map(tuple, pairs))
    opt = solve(rows).total
    obj = sum((rows[i][j] for i, j in pairs), 0)
    return {
        "objective": obj,
        "optimum": opt,
        "gap": obj - opt,
        "proven_optimal": bool(verify_pairs(rows, pairs)),
    }


@dataclass
class ExperimentRecorder:
    """Append-only JSON-serialisable log of sampler runs (config, seed, backend label, energies, verification)."""

    runs: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self, cost: Any, seed: int, sweeps: int, backend: str = "classical-simulated-annealing"
    ) -> dict[str, Any]:
        from munkres._verify import problem_hash

        rows = _as_lists(cost)
        q = assignment_to_qubo(rows)
        bits, e = anneal_qubo(q, seed, sweeps)
        feasible = decode_bits(q, bits)
        repaired = repair_bits(rows, q, bits)
        run = {
            "backend": backend,
            "problem_hash": problem_hash(rows),
            "seed": seed,
            "sweeps": sweeps,
            "variables": len(q.cells),
            "penalty": q.penalty,
            "energy": e,
            "raw_feasible": feasible is not None,
            "repaired": [list(p) for p in repaired],
            "check": verify_candidate(rows, repaired),
        }
        self.runs.append(run)
        return run

    def to_json(self) -> str:
        return json.dumps(self.runs, sort_keys=True, default=str)
