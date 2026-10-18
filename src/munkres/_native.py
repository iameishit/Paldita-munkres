# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Optional native (C) dense kernel: the compiled extension `munkres._lapc`.

It is a pure accelerator. If the extension is missing, fails to import, or the problem is
outside its numeric contract, callers fall back to the pure-Python solver, which gives
identical optimal objectives. Set `PALDITA_NO_NATIVE=1` to disable it.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import threading
from array import array
from collections.abc import Sequence
from typing import Any

__all__ = [
    "FAILED",
    "available",
    "forced_off",
    "native_status",
    "reset_native",
    "set_strict_mode",
    "solve_array",
    "solve_dense",
    "suspended",
    "version",
]

_ABI = 2
_INT64_MAX = (1 << 63) - 1
_SAFE_INT = 1 << 62
_BREAKER_LIMIT = (
    3  # unexpected native failures tolerated before the kernel is switched off for the process
)

_log = logging.getLogger("munkres")
_lock = threading.Lock()
FAILED: Any = object()  # sentinel returned by `safe_call` when the native layer raised
_suppress: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "munkres_native_suspended", default=False
)


class suspended:  # noqa: N801  (kept for the public native API)
    """
    Run a block without the native kernel (this thread / async task only). Used after a native failure so the
    rest of the *same call* goes straight to the exact pure-Python solver instead of retrying the broken kernel
    through another path, which keeps one failing call counted as one failure.
    """

    def __enter__(self) -> None:
        self._token = _suppress.set(True)

    def __exit__(self, *exc: object) -> None:
        _suppress.reset(self._token)


_mod: Any = None
_tried = False
_failures = 0
_recent: list[str] = []
_warned: set[str] = set()
_disabled_reason: str | None = None
_strict = bool(os.environ.get("PALDITA_STRICT"))


def _selftest(mod: Any) -> str | None:
    """Tiny known problems through every entry point; returns a reason string if anything is off."""
    try:
        for kind, code, cost in ((0, "d", [4.0, 1.0, 2.0, 3.0]), (1, "q", [4, 1, 2, 3])):
            r2c = array("i", [-1, -1])
            hall = array("i", bytes(4 * 8))
            rc = mod.solve(kind, 2, 2, array(code, cost), r2c, None, None, hall)
            if rc != 0 or list(r2c) != [1, 0]:
                return f"self-test returned a wrong result (kind {kind}, code {rc})"
            out, status = array("i", [-1, -1]), array("i", [-1])
            if (
                mod.solve_batch(1, 2, 2, array(code, cost), out, status, kind) != 0
                or list(out) != [1, 0]
                or status[0] != 0
            ):
                return f"batch self-test returned a wrong result (kind {kind})"
        return None
    except Exception as exc:
        return f"self-test raised {type(exc).__name__}: {exc}"


def _load() -> Any:
    global _mod, _tried, _disabled_reason
    if _suppress.get():
        return None
    if _tried:
        return _mod
    _tried = True
    if os.environ.get("PALDITA_NO_NATIVE"):
        _disabled_reason = "disabled by PALDITA_NO_NATIVE"
        return None
    try:
        from munkres import _lapc  # type: ignore[attr-defined]

        if _lapc.abi_version() != _ABI:
            _disabled_reason = f"kernel ABI {_lapc.abi_version()} does not match expected {_ABI}; rebuild the extension"
        elif (bad := _selftest(_lapc)) is not None:
            _disabled_reason = bad
            _log.warning("munkres: native kernel rejected (%s); using the pure-Python solver", bad)
        else:
            _mod = _lapc
    except ImportError:
        _disabled_reason = "extension not built (pure-Python solver in use)"
    except Exception as exc:
        _disabled_reason = f"extension failed to load: {type(exc).__name__}: {exc}"
        _log.warning("munkres: %s; using the pure-Python solver", _disabled_reason)
    return _mod


def report_failure(where: str, exc: BaseException) -> None:
    """
    Record an unexpected native failure. Normally this only logs (once per call site) and counts: the
    caller then falls back to the exact pure-Python solver. After `_BREAKER_LIMIT` failures the kernel is
    switched off for the process (circuit breaker). In strict mode the original exception is re-raised.
    """
    global _mod, _failures, _disabled_reason
    if _strict:
        raise exc
    with _lock:
        _failures += 1
        _recent.append(f"{where}: {type(exc).__name__}: {exc}")
        del _recent[:-5]
        first = where not in _warned
        _warned.add(where)
        trip = _mod is not None and _failures >= _BREAKER_LIMIT
        if trip:
            _mod = None
            _disabled_reason = (
                f"circuit breaker open after {_failures} native failures (last: {_recent[-1]})"
            )
    if first:
        _log.warning(
            "munkres: native call %r failed (%s: %s); falling back to the pure-Python solver",
            where,
            type(exc).__name__,
            exc,
        )
    if trip:
        _log.error(
            "munkres: native kernel disabled for this process (%s); call munkres.reset_native() to retry",
            _disabled_reason,
        )


def safe_call(where: str, fn: Any, *args: Any) -> Any:
    """Call a native function; on any unexpected exception report it and return `FAILED`."""
    try:
        return fn(*args)
    except Exception as exc:
        report_failure(where, exc)
        return FAILED


def native_status() -> dict[str, Any]:
    """
    Health of the native accelerator, for logs and health checks: whether it is active, why not, and how
    many unexpected failures have been absorbed. Safe to call at any time; never raises.
    """
    _load()
    return {
        "available": _mod is not None,
        "version": version(),
        "disabled_reason": None if _mod is not None else _disabled_reason,
        "failures": _failures,
        "recent_failures": list(_recent),
        "breaker_limit": _BREAKER_LIMIT,
        "strict": _strict,
    }


def reset_native() -> bool:
    """Clear the failure counters and the circuit breaker, reload the kernel; returns whether it is active."""
    global _mod, _tried, _failures, _disabled_reason
    with _lock:
        _mod, _tried, _failures, _disabled_reason = None, False, 0, None
        _recent.clear()
        _warned.clear()
    return _load() is not None


def set_strict_mode(strict: bool = True) -> None:
    """In strict mode unexpected native failures raise instead of falling back (use in CI, not production)."""
    global _strict
    _strict = bool(strict)


class forced_off:  # noqa: N801
    """Context manager: run the enclosed code on the pure-Python solver (used by cross-backend checks)."""

    def __enter__(self) -> None:
        global _mod, _tried
        self._saved = (_mod, _tried)
        _mod, _tried = None, True

    def __exit__(self, *exc: object) -> None:
        global _mod, _tried
        _mod, _tried = self._saved


def available() -> bool:
    """True if the native kernel loaded."""
    return _load() is not None


def version() -> str:
    return f"c-kernel-abi{_ABI}" if available() else "none"


def _raise_for(rc: int, n: int, hall: Any) -> None:
    if rc == 2:
        raise ValueError("cost matrix contains NaN or -infinity")
    if rc == 1:
        from munkres._core import _Infeasible

        nr, nc = int(hall[0]), int(hall[1])
        raise _Infeasible(
            sorted(int(x) for x in hall[2 : 2 + nr]),
            sorted(int(x) for x in hall[3 + n : 3 + n + nc]),
        )


def solve_dense(  # noqa: PLR0911
    n: int,
    m: int,
    edges: Sequence[Sequence[tuple[int, Any]]],
) -> list[tuple[int, int]] | None:
    """
    Solve the n x m (n <= m) problem given as allowed-edge lists. Returns the `(row, col)` pairs,
    or `None` when the native path does not apply (extension missing, non-int/float costs, range
    too large). Raises `_Infeasible` with a Hall witness when no assignment exists.
    """
    mod = _load()
    if mod is None or n == 0:
        return None
    all_int = True
    flat: Any
    biggest = 0
    for allowed in edges:
        for _, c in allowed:
            t = type(c)
            if t is int or t is bool:
                a = -c if c < 0 else int(c)
                biggest = max(biggest, a)
            elif t is float:
                all_int = False
            else:
                return None
    if all_int:
        if biggest * (n + m + 4) >= _SAFE_INT:
            return None
        flat = array("q", [_INT64_MAX]) * (n * m)
        kind = 1
    else:
        if biggest > (1 << 53):
            return None
        flat = array("d", [float("inf")]) * (n * m)
        kind = 0
    for i, allowed in enumerate(edges):
        base = i * m
        for j, c in allowed:
            flat[base + j] = c
    r2c = array("i", [-1]) * n
    hall = array("i", bytes(4 * (n + m + 4)))
    rc = safe_call("solve", mod.solve, kind, n, m, flat, r2c, None, None, hall)
    if rc is FAILED:
        return None  # soft fallback: the caller uses the exact pure-Python solver
    _raise_for(rc, n, hall)
    if rc != 0:
        return None  # rc 4 (range too large) or allocation failure: caller uses exact arithmetic
    return [(i, r2c[i]) for i in range(n)]


def solve_array(a: Any, *, want_duals: bool = False) -> tuple[Any, Any] | None:
    """
    Zero-copy path for a C-contiguous 2-D numpy float64/int64 array with rows <= cols.
    Returns `(col_of_row, duals_or_None)`; `None` if unavailable or the kernel declines (rc 4).
    """
    mod = _load()
    if mod is None:
        return None
    import numpy as np  # local: numpy is optional for the package

    n, m = a.shape
    kind: int
    ut: Any
    if a.dtype == np.float64:
        kind, ut = 0, np.float64
    elif a.dtype == np.int64:
        kind, ut = 1, np.int64
    else:
        return None
    ibuf = np.empty(2 * n + m + 4, dtype=np.int32)  # r2c(n) | hall(n+m+4)
    r2c, hall = ibuf[:n], ibuf[n:]
    u = np.empty(n, dtype=ut) if want_duals else None
    v = np.empty(m, dtype=ut) if want_duals else None
    rc = safe_call("solve_array", mod.solve, kind, n, m, a, r2c, u, v, hall)
    if rc is FAILED:
        return FAILED  # type: ignore[no-any-return]  # sentinel: caller must not retry the kernel
    if rc in (1, 2):
        _raise_for(rc, n, hall)
    if rc != 0:
        return None
    return r2c, ((u, v) if want_duals else None)


if __name__ == "__main__":  # pragma: no cover
    print(version(), file=sys.stderr)
