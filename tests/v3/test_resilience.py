"""
Production-safety contract: a broken, missing, miscompiled or misbehaving native kernel must never change
results or break the caller. Every test injects a fault and requires the answer to equal the pure-Python
solver's, then restores the real kernel.
"""

import logging
import os
import random

import pytest

import munkres
from munkres import _dense, _native, linear_sum_assignment, solve, solve_batch

np = pytest.importorskip("numpy")
REAL = _native._load()  # the genuine kernel (None when the extension is not built)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _Boom:
    """A kernel that answers the ABI question and then fails on every real call."""

    def __init__(self, exc=RuntimeError("injected fault")):
        self.exc = exc
        self.calls = 0

    def abi_version(self):
        return _native._ABI

    def _fail(self, *a, **k):
        self.calls += 1
        raise self.exc

    solve = solve_batch = lsa = _fail


class _Fault:
    """Install `mod` as the active kernel (and in the ndarray quick path); always restore the real one."""

    def __init__(self, mod):
        self.mod = mod
        self.log = _Records()

    def __enter__(self):
        _native.reset_native()
        self.saved_state = _dense._STATE
        _native._mod, _native._tried = self.mod, True
        st = _dense._init()
        if st:
            lst = list(st)
            lst[4], lst[8] = getattr(self.mod, "lsa", None), self.mod
            _dense._STATE = tuple(lst)
        logging.getLogger("munkres").addHandler(self.log)
        return self

    def __exit__(self, *exc):
        logging.getLogger("munkres").removeHandler(self.log)
        _dense._STATE = None
        _native.set_strict_mode(False)
        _native.reset_native()


def _matrix(seed=0, n=6):
    return np.random.default_rng(seed).random((n, n))


def test_results_are_identical_when_the_kernel_raises_on_every_call():
    a = _matrix()
    want = a[linear_sum_assignment(a)].sum()
    with _Fault(_Boom()):
        r, c = linear_sum_assignment(a)
        assert a[r, c].sum() == pytest.approx(want)
        assert solve(a.tolist()).total == pytest.approx(want)
        b = np.random.default_rng(1).random((20, 5, 5))
        res = solve_batch(b)
        assert res.status.tolist() == [0] * 20
        for k in range(20):
            assert b[k][np.arange(5), res.col_ind[k]].sum() == pytest.approx(
                solve(b[k].tolist()).total
            )


def test_each_failing_call_site_is_logged_exactly_once_and_nothing_is_raised():
    a = _matrix(2)
    with _Fault(_Boom()) as f:
        for _ in range(
            2
        ):  # one call reaches two native sites: the quick path, then the general kernel path
            linear_sum_assignment(a)
        warnings = [m for m in f.log.messages if "falling back" in m]
        assert len(warnings) == len(set(warnings)) >= 1, "a call site must not be logged twice"
        assert any("'lsa'" in m for m in warnings)
        assert _native.native_status()["failures"] >= 2


def test_circuit_breaker_opens_after_repeated_failures_then_stops_calling_the_kernel():
    boom = _Boom()
    with _Fault(boom) as f:
        for seed in range(10):
            linear_sum_assignment(_matrix(seed))
        status = _native.native_status()
        assert status["available"] is False and "circuit breaker" in status["disabled_reason"]
        assert status["failures"] == _native._BREAKER_LIMIT
        calls = boom.calls
        linear_sum_assignment(_matrix(99))
        assert boom.calls == calls, "an open breaker must not keep calling the broken kernel"
        assert any("disabled for this process" in m for m in f.log.messages)


def test_reset_native_closes_the_breaker():
    with _Fault(_Boom()):
        for seed in range(5):
            linear_sum_assignment(_matrix(seed))
        assert not munkres.native_status()["available"]
    status = munkres.native_status()  # _Fault restored the real kernel on exit
    assert status["available"] == (REAL is not None)
    assert status["failures"] == 0 and status["recent_failures"] == []
    assert munkres.reset_native() == (REAL is not None)


def test_strict_mode_reraises_instead_of_falling_back():
    with _Fault(_Boom(ValueError("strict boom"))):
        munkres.set_strict_mode(True)
        with pytest.raises(ValueError, match="strict boom"):
            linear_sum_assignment(_matrix())


def test_user_errors_are_still_reported_not_swallowed_as_kernel_faults():
    with _Fault(_Boom()):
        with pytest.raises(munkres.UnsolvableMatrix):
            linear_sum_assignment(np.array([[np.inf, np.inf], [1.0, 2.0]]))
        with pytest.raises(ValueError):
            linear_sum_assignment(np.array([[np.nan, 1.0], [1.0, 2.0]]))


class _WrongKernel(_Boom):
    """Loads fine, claims success, returns garbage: only the load-time self-test can catch this."""

    def solve(self, kind, n, m, cost, r2c, u, v, hall):
        for i in range(n):
            r2c[i] = 0
        return 0

    def solve_batch(self, *a):
        return 0


def test_selftest_rejects_a_kernel_that_returns_wrong_answers():
    assert _native._selftest(_WrongKernel()) is not None
    assert _native._selftest(_Boom()) is not None
    if REAL is not None:
        assert _native._selftest(REAL) is None, "the shipped kernel must pass its own self-test"


def test_incompatible_extension_degrades_to_pure_python():
    import sys

    if os.environ.get("PALDITA_NO_NATIVE"):
        pytest.skip("the loader is bypassed entirely when PALDITA_NO_NATIVE is set")

    class _OldAbi(_Boom):
        def abi_version(self):
            return _native._ABI + 1

    a = _matrix(4)
    want = a[linear_sum_assignment(a)].sum()
    saved_mod = sys.modules.get("munkres._lapc")
    saved_attr = getattr(munkres, "_lapc", None)
    sys.modules["munkres._lapc"] = munkres._lapc = _OldAbi()
    try:
        _native._mod, _native._tried, _native._disabled_reason = None, False, None
        _dense._STATE = None
        assert _native._load() is None
        assert "ABI" in munkres.native_status()["disabled_reason"]
        r, c = linear_sum_assignment(a)
        assert a[r, c].sum() == pytest.approx(want)
    finally:
        if saved_mod is not None:
            sys.modules["munkres._lapc"] = saved_mod
        else:
            sys.modules.pop("munkres._lapc", None)
        if saved_attr is not None:
            munkres._lapc = saved_attr
        elif hasattr(munkres, "_lapc"):
            del munkres._lapc
        _dense._STATE = None
        _native.reset_native()
    assert munkres.native_status()["available"] == (REAL is not None)


def test_forced_pure_python_gives_the_same_answers():
    rng = random.Random(3)
    rows = [[rng.randint(0, 9) for _ in range(5)] for _ in range(5)]
    want = solve(rows).total
    with _native.forced_off():
        assert solve(rows).total == want
        assert linear_sum_assignment(np.array(rows))[0].tolist() == [0, 1, 2, 3, 4]
        assert munkres.native_status()["available"] is False


def test_native_status_is_always_safe_to_call_and_json_friendly():
    import json

    status = munkres.native_status()
    json.dumps(status)
    assert set(status) >= {
        "available",
        "version",
        "disabled_reason",
        "failures",
        "recent_failures",
        "breaker_limit",
        "strict",
    }
