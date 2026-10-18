# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Batched solving: many same-shape problems in a single native call (amortises call overhead)."""

from __future__ import annotations

import contextlib
from typing import Any

from munkres import _native
from munkres._core import UnsolvableMatrix

__all__ = ["BatchResult", "solve_batch"]


class BatchResult:
    """`col_ind[b, i]` is the column of row i in problem b (or -1 if that problem failed)."""

    def __init__(self, col_ind: Any, status: Any) -> None:
        self.col_ind = col_ind
        self.status = status  # 0 ok, 1 infeasible, 2 NaN/-inf, 4 fallback needed

    @property
    def ok(self) -> Any:
        return self.status == 0


_FLOAT_EXACT = 1 << 53  # every integer with |x| <= 2**53 is exactly representable as float64
_ON_INFEASIBLE = ("raise", "mark")


def _lossy_problems(orig: Any, np: Any) -> Any:
    """
    Boolean mask (one entry per problem): True where converting the problem to float64 would
    change a value, i.e. where the float64 kernel cannot be trusted and an exact path is needed.
    """
    b = orig.shape[0]
    kind = orig.dtype.kind
    if kind == "b" or b == 0 or orig.size == 0:
        return np.zeros(b, dtype=bool)
    flat = orig.reshape(b, -1)
    if kind == "u":
        return np.asarray((flat > _FLOAT_EXACT).any(axis=1))
    if kind == "i":
        return np.asarray(((flat > _FLOAT_EXACT) | (flat < -_FLOAT_EXACT)).any(axis=1))
    if kind == "f":
        if orig.dtype.itemsize <= 8:
            return np.zeros(b, dtype=bool)
        with np.errstate(all="ignore"):  # extended precision: lossless only if the round trip is
            back = flat.astype(np.float64).astype(orig.dtype)
            return np.asarray(((back != flat) & ~np.isnan(flat)).any(axis=1))
    return np.ones(b, dtype=bool)


def solve_batch(costs: Any, maximize: bool = False, on_infeasible: str = "raise") -> BatchResult:
    """
    Solve `costs` of shape (B, n, m) with n <= m (each problem independent).
    `on_infeasible="raise"` raises `UnsolvableMatrix` naming the first failing index;
    `"mark"` returns status codes instead.

    Exactness contract: the input dtype is never silently narrowed. float16/32/64, bool and
    integers with |x| <= 2**53 run on the float64 kernel (exactly representable). Any problem
    holding larger integers, extended-precision floats that do not round-trip through float64,
    or object entries (Fraction, Decimal, ...) is solved exactly: first by the int64 kernel when
    its range allows, otherwise by the general exact solver.
    """
    import numpy as np

    if on_infeasible not in _ON_INFEASIBLE:
        raise ValueError(f"on_infeasible must be one of {_ON_INFEASIBLE}, got {on_infeasible!r}")
    orig = np.asarray(costs)
    if orig.ndim != 3:
        raise ValueError("costs must have shape (batch, rows, cols)")
    if orig.dtype.kind not in "biufO":
        raise TypeError(f"unsupported cost dtype {orig.dtype}")
    b, n, m = orig.shape
    if n > m:
        raise ValueError("batch solver requires rows <= cols; pass the transpose")
    mod = _native._load()
    kernel_failed = False
    out = np.full((b, n), -1, dtype=np.int32)
    status = np.zeros(b, dtype=np.int32)
    if b and n:
        if mod is None or orig.dtype.kind == "O":
            status[:] = 4
        elif orig.dtype.kind in "iub":
            # exact int64 kernel. INT64_MAX is the kernel's "forbidden" marker, so every problem is
            # range-guarded first: out-of-range problems are blanked here and solved exactly below.
            bound = 3 * 10**18 // (n + m + 4)
            flat = orig.reshape(b, -1)
            unsafe = np.asarray((flat > bound).any(axis=1))
            if orig.dtype.kind == "i":
                unsafe |= np.asarray((flat < -bound).any(axis=1))
            a = np.zeros(orig.shape, dtype=np.int64)
            ok = ~unsafe
            a[ok] = orig[ok]
            if maximize:
                a = -a
            done = _native.safe_call(
                "solve_batch", mod.solve_batch, b, n, m, np.ascontiguousarray(a), out, status, 1
            )
            if done is _native.FAILED or done != 0:
                kernel_failed = True
                status[:] = 4  # soft fallback: every problem is solved by the exact Python solver
            else:
                status[unsafe] = 4
        else:
            a = np.ascontiguousarray(orig, dtype=np.float64)
            if maximize:
                a = -a
            done = _native.safe_call("solve_batch", mod.solve_batch, b, n, m, a, out, status)
            if done is _native.FAILED or done != 0:
                kernel_failed = True
                status[:] = 4  # soft fallback: every problem is solved by the exact Python solver
            else:
                status[_lossy_problems(orig, np)] = 4  # kernel result is untrustworthy for these
    redo = np.flatnonzero(status == 4)
    # if the kernel itself failed, the exact re-solves must not touch it again (one failure, not B)
    with _native.suspended() if kernel_failed else contextlib.nullcontext():
        for k in redo:
            out[k] = -1
            status[k] = _solve_exact(orig[k], maximize, out[k])
    bad = np.flatnonzero(status != 0)
    out[bad] = -1
    if bad.size and on_infeasible == "raise":
        code = int(status[bad[0]])
        what = {1: "infeasible", 2: "contains NaN or -inf"}.get(code, f"failed (code {code})")
        raise UnsolvableMatrix(f"problem {int(bad[0])} is {what}")
    return BatchResult(out, status)


def _solve_exact(problem: Any, maximize: bool, row_out: Any) -> int:
    """Exactly solve one (n, m) problem into `row_out`; returns the status code (0 ok)."""
    from munkres._api import solve

    if problem.dtype.kind in "iu":
        from munkres._dense import fast_lsa

        try:
            got = fast_lsa(problem, maximize)  # exact int64 kernel when the range is safe
        except UnsolvableMatrix:
            return 1
        if got is not None:
            row_out[:] = got[1]
            return 0
    try:
        res = solve(problem.tolist(), maximize=maximize)  # python ints / exact scalars
    except UnsolvableMatrix:
        return 1
    except (ValueError, TypeError):
        return 2
    row_out[:] = [j for _, j in sorted(res.pairs)]
    return 0
