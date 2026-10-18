# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Exact counting and an explicit Gibbs model over assignments (small problems).

Model (stated, not assumed): P(M) = exp(-beta * cost(M)) / Z over all injections M of the rows
into distinct columns using allowed cells. `edge_marginals` are P((i, j) in M) under that model:
they are *not* confidence scores for an optimiser's answer. Exact subset DP, O(n * 2^m * m):
limited to m <= 20 columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from munkres._core import _as_lists, _validated_edges

__all__ = ["Gibbs", "count_matchings", "gibbs", "gibbs_sample"]

MAX_COLS = 20


def _prep(matrix: Any) -> tuple[int, int, list[dict[int, Any]]]:
    n, m, edges = _validated_edges(_as_lists(matrix))
    if n > m:
        raise ValueError("requires rows <= cols; pass the transpose")
    if m > MAX_COLS:
        raise ValueError(f"exact DP supports at most {MAX_COLS} columns")
    return n, m, [dict(e) for e in edges]


def count_matchings(matrix: Any) -> int:
    """Exact number of ways to match every row to a distinct allowed column (a 0/1 permanent for square)."""
    n, _m, rows = _prep(matrix)
    dp = {0: 1}
    for i in range(n):
        nxt: dict[int, int] = {}
        for mask, w in dp.items():
            for j in rows[i]:
                if not mask >> j & 1:
                    nxt[mask | 1 << j] = nxt.get(mask | 1 << j, 0) + w
        dp = nxt
    return sum(dp.values())


@dataclass(frozen=True)
class Gibbs:
    beta: float
    log_z: float
    marginals: list[list[float]]  # marginals[i][j] = P((i, j) in M); 0 for forbidden cells
    expected_cost: float
    entropy: float  # Shannon entropy (nats) of the distribution over assignments
    n_matchings: int


_NEG_INF = float("-inf")


def _check_beta(beta: Any) -> float:
    """`beta` must be a finite real number (0 gives the uniform distribution, negative favours high cost)."""
    try:
        b = float(beta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"beta must be a finite real number, got {beta!r}") from None
    if not math.isfinite(b):
        raise ValueError(f"beta must be finite, got {beta!r}")
    return b


def _fcost(c: Any) -> float:
    try:
        x = float(c)
    except OverflowError:
        raise ValueError("cost is too large for the floating-point Gibbs model") from None
    if not math.isfinite(x):
        raise ValueError(
            "cost must be finite for the Gibbs model (use DISALLOWED for forbidden cells)"
        )
    return x


def _lse(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) without leaving log space (so nothing underflows or overflows)."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _log_weights(rows: list[dict[int, Any]], beta: float) -> tuple[list[dict[int, float]], float]:
    """Log weights with each row shifted by its minimum cost (better conditioned); returns also the total shift."""
    costs = [{j: _fcost(c) for j, c in r.items()} for r in rows]
    shifts = [min(r.values()) if r else 0.0 for r in costs]
    lw = [{j: -beta * (c - shifts[i]) for j, c in r.items()} for i, r in enumerate(costs)]
    return lw, sum(shifts)


def gibbs(matrix: Any, beta: float = 1.0) -> Gibbs:
    """
    Exact log partition function, edge marginals, expected cost and entropy of the Gibbs model above.

    The dynamic programme runs entirely in log space: a feasible problem is never reported as
    infeasible because a probability weight underflowed, whatever `beta` and the cost range are.
    """
    beta = _check_beta(beta)
    n, m, rows = _prep(matrix)
    if n == 0:
        return Gibbs(beta, 0.0, [], 0.0, 0.0, 1)
    lw, shift_total = _log_weights(rows, beta)
    fwd: list[dict[int, float]] = [{0: 0.0}]
    for i in range(n):
        nxt: dict[int, float] = {}
        for mask, f in fwd[i].items():
            for j, wij in lw[i].items():
                if not mask >> j & 1:
                    key = mask | 1 << j
                    nxt[key] = _lse(nxt.get(key, _NEG_INF), f + wij)
        fwd.append(nxt)
    if not fwd[n]:  # structural infeasibility: no mask is reachable (not an underflow artefact)
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible assignment")
    ref = max(
        fwd[n].values()
    )  # normalise against the largest path weight: the (possibly huge) offset
    log_z_rel = math.log(sum(math.exp(v - ref) for v in fwd[n].values()))  # never enters the ratios
    log_z = ref + log_z_rel
    memo: dict[tuple[int, int], float] = {}

    def back(k: int, mask: int) -> float:
        if k == n:
            return 0.0
        key = (k, mask)
        got = memo.get(key)
        if got is None:
            got = _NEG_INF
            for j, wkj in lw[k].items():
                if not mask >> j & 1:
                    got = _lse(got, wkj + back(k + 1, mask | 1 << j))
            memo[key] = got
        return got

    logm = [[_NEG_INF] * m for _ in range(n)]
    for i in range(n):
        for mask, f in fwd[i].items():
            for j, wij in lw[i].items():
                if not mask >> j & 1:
                    logm[i][j] = _lse(logm[i][j], f + wij + back(i + 1, mask | 1 << j))
    marg = [
        [min(1.0, math.exp((x - ref) - log_z_rel)) if x != _NEG_INF else 0.0 for x in r]
        for r in logm
    ]
    exp_cost = sum(marg[i][j] * _fcost(c) for i, r in enumerate(rows) for j, c in r.items())
    log_z -= beta * shift_total  # undo the row shifts: this is log Z of the unshifted model
    return Gibbs(beta, log_z, marg, exp_cost, log_z + beta * exp_cost, count_matchings(matrix))


def gibbs_sample(matrix: Any, beta: float = 1.0, rng: Any = None) -> tuple[tuple[int, int], ...]:
    """One exact sample of an assignment from the Gibbs model of `gibbs` (sequential conditional sampling, log space)."""
    import random

    beta = _check_beta(beta)
    rng = rng or random.Random()  # noqa: S311  # nosec B311
    n, _m, rows = _prep(matrix)
    lw, _ = _log_weights(rows, beta)
    memo: dict[tuple[int, int], float] = {}

    def back(k: int, mask: int) -> float:
        if k == n:
            return 0.0
        key = (k, mask)
        got = memo.get(key)
        if got is None:
            got = _NEG_INF
            for j, wkj in lw[k].items():
                if not mask >> j & 1:
                    got = _lse(got, wkj + back(k + 1, mask | 1 << j))
            memo[key] = got
        return got

    if back(0, 0) == _NEG_INF:
        from munkres._core import UnsolvableMatrix

        raise UnsolvableMatrix("no feasible assignment")
    mask, pairs = 0, []
    for i in range(n):
        logs = [
            (j, wij + back(i + 1, mask | 1 << j)) for j, wij in lw[i].items() if not mask >> j & 1
        ]
        top = max(x for _, x in logs)
        opts = [(j, math.exp(x - top) if x != _NEG_INF else 0.0) for j, x in logs]
        r = rng.random() * sum(x for _, x in opts)
        acc = 0.0
        pick = next(j for j, x in reversed(opts) if x > 0)
        for j, x in opts:
            acc += x
            if r < acc and x > 0:
                pick = j
                break
        pairs.append((i, pick))
        mask |= 1 << pick
    return tuple(pairs)
