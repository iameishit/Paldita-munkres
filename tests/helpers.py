"""
Shared test helpers: assignment validation, independent optimality oracles,
and a hang guard.

The oracles deliberately do NOT use the code under test:
  * brute_force_min_cost -- exhaustive search (tiny matrices only)
  * scipy_min_cost       -- scipy.optimize.linear_sum_assignment (any size)
"""

import itertools
import signal
import threading

from munkres import DISALLOWED

# ---------------------------------------------------------------------------
# Hang guard
# ---------------------------------------------------------------------------


class Hang(Exception):
    """The guarded call did not finish in time (== infinite loop)."""


def call_with_timeout(func, seconds=2, *args, **kwargs):
    """
    Call func(*args, **kwargs); raise Hang if it takes longer than `seconds`.
    Uses SIGALRM where available (POSIX); otherwise a daemon thread.
    """
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():

        def _handler(signum, frame):
            raise Hang("call exceeded {}s".format(seconds))

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    box = {}

    def _run():
        try:
            box["value"] = func(*args, **kwargs)
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise Hang("call exceeded {}s".format(seconds))
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def shape(matrix):
    return len(matrix), len(matrix[0])


def assert_valid_assignment(matrix, indices):
    """A correct result has min(rows, cols) unique in-range, allowed pairs."""
    rows, cols = shape(matrix)
    assert len(indices) == min(rows, cols), "expected {} pairs, got {}: {}".format(
        min(rows, cols), len(indices), indices
    )
    assert len({r for r, _ in indices}) == len(indices), "row used twice"
    assert len({c for _, c in indices}) == len(indices), "column used twice"
    for r, c in indices:
        assert 0 <= r < rows and 0 <= c < cols, "({}, {}) outside {}x{}".format(r, c, rows, cols)
        assert matrix[r][c] is not DISALLOWED, "({}, {}) is DISALLOWED".format(r, c)


def total(matrix, indices):
    return sum(matrix[r][c] for r, c in indices)


# ---------------------------------------------------------------------------
# Oracles. Both return the optimal total cost, or None if no complete
# assignment (every row of the smaller side matched) exists.
# ---------------------------------------------------------------------------


def _transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def brute_force_min_cost(matrix):
    rows, cols = shape(matrix)
    if rows > cols:
        matrix = _transpose(matrix)
        rows, cols = cols, rows
    best = None
    for chosen in itertools.permutations(range(cols), rows):
        cost = 0
        for r, c in enumerate(chosen):
            v = matrix[r][c]
            if v is DISALLOWED:
                break
            cost += v
        else:
            if best is None or cost < best:
                best = cost
    return best


def scipy_min_cost(matrix):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    finite = [abs(v) for row in matrix for v in row if v is not DISALLOWED]
    big = (sum(finite) + 1) * (max(shape(matrix)) + 1)
    arr = np.array([[big if v is DISALLOWED else v for v in row] for row in matrix], dtype=float)
    ri, ci = linear_sum_assignment(arr)
    if any(matrix[r][c] is DISALLOWED for r, c in zip(ri, ci)):
        return None
    return sum(matrix[r][c] for r, c in zip(ri, ci))


def all_matchings(matrix):
    """Every complete matching of the smaller side, as (total cost, pairs). Exhaustive."""
    rows, cols = shape(matrix)
    flip = rows > cols
    grid = _transpose(matrix) if flip else matrix
    n, m = len(grid), len(grid[0])
    for chosen in itertools.permutations(range(m), n):
        if any(grid[r][c] is DISALLOWED for r, c in enumerate(chosen)):
            continue
        pairs = [(c, r) if flip else (r, c) for r, c in enumerate(chosen)]
        yield sum(grid[r][c] for r, c in enumerate(chosen)), sorted(pairs)
