# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Array-in / array-out fast path (NumPy ndarray), SciPy-compatible results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from munkres import _native
from munkres._core import UnsolvableMatrix

if TYPE_CHECKING:
    from munkres._core import _Infeasible

_SAFE = 1 << 62


def _fits_float64(a: Any, np: Any) -> bool:
    """True if every value of the extended-precision array survives a float64 round trip unchanged."""
    with np.errstate(all="ignore"):
        back = a.astype(np.float64).astype(a.dtype)
        return bool(np.all((back == a) | np.isnan(a)))


_AR_MAX = 4096
_STATE: Any = (
    None  # None = not initialised yet, False = fast path unavailable, else a tuple (see _init)
)


def _init() -> Any:
    """Resolve numpy, dtypes, the native entry point and the cached index vector once."""
    global _STATE
    try:
        import numpy as np
    except ImportError:
        _STATE = False
        return False
    mod = _native._load()
    lsa = getattr(mod, "lsa", None)
    if lsa is None or np.dtype(np.intp).itemsize != 8:
        _STATE = False
        return False
    ar = np.arange(_AR_MAX, dtype=np.intp)
    ar.flags.writeable = False  # shared: callers only ever see copies
    _STATE = (
        np.ndarray,
        np.dtype(np.float64),
        np.dtype(np.int64),
        np.intp,
        lsa,
        ar,
        np.empty,
        np.arange,
        mod,
    )
    return _STATE


def _quick(cost: Any, maximize: bool) -> Any:  # noqa: PLR0911
    """
    Hot path for the common case: a C-contiguous 2-D float64/int64 ndarray with rows <= cols.
    One kernel call writes the result straight into a fresh intp array. Returns None whenever
    anything is unusual (other dtype/layout, infeasible, NaN, range) so `fast_lsa` handles it with full
    diagnostics - the quick path never changes behaviour, it only skips work.
    """
    st = _STATE if _STATE is not None else _init()
    if st is False or _native._mod is not st[8]:  # kernel switched off (breaker, forced_off, reset)
        return None
    if type(cost) is not st[0] or cost.ndim != 2 or not cost.flags.c_contiguous:
        return None
    dt = cost.dtype
    if dt is st[1]:
        kind = 0
    elif dt is st[2]:
        kind = 1
    else:
        return None
    n, m = cost.shape
    if n > m or n == 0:
        return None
    cols = st[6](n, st[3])
    try:
        rc = st[4](kind, n, m, cost, cols, maximize)
    except Exception as exc:
        _native.report_failure("lsa", exc)
        return _native.FAILED
    if rc != 0:
        return None
    return (st[5][:n].copy() if n <= _AR_MAX else st[7](n, dtype=st[3])), cols


def fast_lsa(cost: Any, maximize: bool = False) -> Any:  # noqa: PLR0911
    """
    `linear_sum_assignment` for a 2-D numpy array of float or integer dtype.

    Returns `(row_ind, col_ind)` or `None` when this path does not apply (kernel
    missing, unsupported dtype, int range too large) so the caller can use the
    general solver. Infeasible problems raise `UnsolvableMatrix`.
    """
    got = _quick(cost, maximize)
    if got is not None:  # a result, or `FAILED` (the kernel raised: the caller must not retry it)
        return got
    import numpy as np

    from munkres._core import _explain, _Infeasible

    a = cost
    if not isinstance(a, np.ndarray) or a.ndim != 2 or not _native.available():
        return None
    kind = a.dtype.kind
    if kind == "b":
        a = a.astype(np.int64)
    elif kind in "iu":
        if a.size and (
            a.max() > _SAFE // (sum(a.shape) + 4) or a.min() < -(_SAFE // (sum(a.shape) + 4))
        ):
            return None
        a = a.astype(np.int64, copy=False)
    elif kind == "f":
        if a.dtype.itemsize > 8 and not _fits_float64(a, np):
            return None  # extended precision that float64 would round: use the exact general solver
        a = a.astype(np.float64, copy=False)  # overflow range is checked in the kernel (rc 4)
    else:
        return None
    n, m = a.shape
    if n == 0 or m == 0:
        e = np.empty(0, dtype=np.intp)
        return e, e.copy()
    if maximize:
        a = -a  # int64 range was bounded above; -inf (forbidden when maximising) -> +inf
    transposed = n > m
    work = np.ascontiguousarray(a.T if transposed else a)
    try:
        got = _native.solve_array(work)
    except _Infeasible as bad:
        left, right = ("column", "row") if transposed else ("row", "column")
        raise _explain(bad, left, right, swapped=transposed) from None
    if got is None:
        return None
    r2c = got[0].astype(np.intp)
    if transposed:  # work rows are original columns
        order = np.argsort(r2c, kind="stable")
        return r2c[order], order.astype(np.intp)
    return np.arange(n, dtype=np.intp), r2c


__all__ = ["UnsolvableMatrix", "fast_lsa"]


class AssignmentSolver:
    """
    Reusable solver for a fixed shape/dtype: scratch buffers are allocated once and reused,
    which removes per-call allocation for repeated solves of same-shaped matrices.

        solver = AssignmentSolver((256, 256))
        col = solver.solve(A)      # column of each row; rows <= cols required
    """

    def __init__(self, shape: tuple[int, int], dtype: str = "float64") -> None:
        import numpy as np

        n, m = shape
        if n > m:
            raise ValueError("AssignmentSolver requires rows <= cols; pass the transpose")
        if dtype not in ("float64", "int64"):
            raise ValueError("dtype must be 'float64' or 'int64'")
        self.shape, self.dtype = (n, m), np.dtype(dtype)
        self._ibuf = np.empty(2 * n + m + 4, dtype=np.int32)

    def solve(self, cost: Any, maximize: bool = False) -> Any:
        import numpy as np

        from munkres._core import _explain

        a = np.asarray(cost)
        if a.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {a.shape}")
        a = np.ascontiguousarray(-a if maximize else a, dtype=self.dtype)
        mod = _native._load()
        n, m = self.shape
        if mod is None:
            from munkres._api import solve

            with _native.suspended():  # the kernel just failed: finish this call on pure Python
                return np.array([j for _, j in sorted(solve(a).pairs)], dtype=np.intp)
        ib = self._ibuf
        rc = _native.safe_call(
            "solver",
            mod.solve,
            0 if self.dtype.kind == "f" else 1,
            n,
            m,
            a,
            ib[:n],
            None,
            None,
            ib[n:],
        )
        if rc is _native.FAILED:
            from munkres._api import solve

            return np.array([j for _, j in sorted(solve(a).pairs)], dtype=np.intp)
        try:
            _native._raise_for(rc, n, ib[n:])
        except _native_infeasible() as bad:
            raise _explain(bad, "row", "column", swapped=False) from None
        if rc != 0:
            from munkres._api import solve

            return np.array([j for _, j in sorted(solve(a).pairs)], dtype=np.intp)
        return ib[:n].astype(np.intp)


def _native_infeasible() -> type[_Infeasible]:
    from munkres._core import _Infeasible

    return _Infeasible
