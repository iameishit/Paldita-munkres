# Munkres -- the Hungarian / Kuhn-Munkres assignment algorithm.
#
# Copyright (c) 2008-2020 Brian M. Clapper (original author)
# Copyright (c) 2026 Eishit Nigam (modifications for 2.x)
#
# Licensed under the Apache License, Version 2.0. See LICENSE.md and NOTICE.
# This file was substantially modified in 2026 (the 2.x core rewrite); see
# CHANGELOG.md for the list of changes.
"""
The solver, validation and helpers behind the public `munkres` API.

Import from `munkres`, not from this private module.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from fractions import Fraction
from numbers import Real
from typing import Any, ClassVar, Protocol, TypeAlias

from munkres._trace import Trace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF = float("inf")
DISALLOWED_PRINTVAL = "D"


class DISALLOWED_OBJ:  # noqa: N801  (public legacy name)
    """
    Sentinel for a forbidden pairing. There is exactly one instance
    (`DISALLOWED`); it survives `copy`, `deepcopy` and `pickle` (every
    protocol) as that very same object, so identity checks keep working
    across processes.
    """

    _instance: ClassVar[DISALLOWED_OBJ | None] = None

    def __new__(cls) -> DISALLOWED_OBJ:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DISALLOWED"

    def __reduce__(self) -> str:
        # A string means "the module-level global with this name".
        return "DISALLOWED"


DISALLOWED = DISALLOWED_OBJ()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Number: TypeAlias = int | float | Decimal | Fraction
"""Any cost the solver can do arithmetic on."""

AnyNum: TypeAlias = Number
"""Legacy name for `Number` (kept for backward compatibility)."""

Cell: TypeAlias = Number | DISALLOWED_OBJ
"""A matrix cell: a cost, or `DISALLOWED`."""

Matrix: TypeAlias = Sequence[Sequence[Cell]]
"""A cost matrix as nested sequences (lists, tuples, ...)."""


class _SupportsToList(Protocol):
    def tolist(self) -> Any: ...  # numpy arrays


class _SupportsToNumpy(Protocol):
    def to_numpy(self) -> Any: ...  # pandas DataFrames


MatrixLike: TypeAlias = Matrix | _SupportsToList | _SupportsToNumpy
"""Anything accepted as a cost matrix: nested sequences, numpy arrays,
pandas DataFrames."""

_NUMBER_TYPES = (Real, Decimal)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnsolvableMatrix(ValueError):  # noqa: N818  (public name, kept for compatibility)
    """
    Raised when no complete assignment exists.

    When the cause is a conflict between forbidden cells, the exception
    carries a proof: `rows` can only be matched to `cols`, and there are more
    `rows` than `cols` (Hall's condition is violated). Both are tuples of
    indexes into the caller's original matrix. `rows`/`cols` are empty when no
    such witness exists.

    It subclasses `ValueError` (like SciPy's "cost matrix is infeasible"), so
    `except ValueError` also catches it.
    """

    def __init__(
        self,
        message: str = "Matrix cannot be solved!",
        rows: Sequence[int] = (),
        cols: Sequence[int] = (),
    ) -> None:
        super().__init__(message)
        self.rows: tuple[int, ...] = tuple(rows)
        self.cols: tuple[int, ...] = tuple(cols)


class _Infeasible(Exception):  # noqa: N818
    """Internal: carries the Hall-violation witness out of the solver."""

    def __init__(self, rows: list[int], cols: list[int]) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _as_lists(matrix: Any) -> list[list[Any]]:
    """
    Return a brand-new list-of-lists copy of `matrix`. Accepts lists, tuples,
    numpy arrays and pandas DataFrames. The caller's object is never touched
    (a numpy row slice is a *view*, which is why the old code could mutate it).
    """
    if hasattr(matrix, "to_numpy"):  # pandas DataFrame
        matrix = matrix.to_numpy()
    if hasattr(matrix, "tolist"):  # numpy array -> plain Python values
        matrix = matrix.tolist()
    try:
        rows: list[list[Any]] = []
        for row in matrix:
            if isinstance(row, (str, bytes)):
                raise TypeError
            rows.append(list(row))
    except TypeError:
        raise TypeError(
            "cost matrix must be a sequence of rows (list, tuple, numpy array "
            "or DataFrame), e.g. [[1, 2], [3, 4]]"
        ) from None
    return rows


def _validated_edges(matrix: Any) -> tuple[int, int, list[list[tuple[int, Any]]]]:
    """
    Validate `matrix` and return `(n_rows, n_cols, edges)` where
    `edges[i]` lists `(column, cost)` for every *allowed* cell of row `i`.

    Rules (all violations raise immediately -- nothing can loop forever):

    - must be rectangular (an empty matrix is fine)  -> `ValueError`
    - cells must be numbers or `DISALLOWED`          -> `TypeError`
    - `NaN` and `-inf` are meaningless as costs      -> `ValueError`
    - `+inf` means "impossible" and is treated exactly like `DISALLOWED`
    """
    rows = _as_lists(matrix)
    if not rows:
        return 0, 0, []
    width = len(rows[0])
    edges: list[list[tuple[int, Any]]] = []
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"cost matrix is not rectangular: row {i} has {len(row)} "
                f"entries but row 0 has {width}"
            )
        allowed: list[tuple[int, Any]] = []
        for j, value in enumerate(row):
            if value is DISALLOWED:
                continue
            if not isinstance(value, _NUMBER_TYPES):
                raise TypeError(f"cell [{i}][{j}] is not a number: {value!r}")
            try:
                is_nan = value != value  # noqa: PLR0124
                is_pos_inf = value == _INF
                is_neg_inf = value == -_INF
            except ArithmeticError:  # Decimal('sNaN') signals on compare
                is_nan = True
            if is_nan:
                raise ValueError(f"cell [{i}][{j}] is NaN")
            if is_pos_inf:
                continue  # +inf == DISALLOWED
            if is_neg_inf:
                raise ValueError(
                    f"cell [{i}][{j}] is -infinity, which has no defined assignment cost"
                )
            allowed.append((j, value))
        edges.append(allowed)
    return len(rows), width, edges


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def _solve(
    n: int,
    m: int,
    edges: Sequence[Sequence[tuple[int, Any]]],
    events: list[dict[str, Any]] | None = None,
    duals: list[Any] | None = None,
) -> list[tuple[int, int]]:
    """
    Minimum-cost assignment of all `n` rows to distinct columns of `m >= n`
    columns, using only the allowed `edges`. Shortest-augmenting-path
    Hungarian method with dual potentials: O(n^2 * m).

    Termination is guaranteed by construction: every pass of the inner loop
    permanently marks one more column as visited, so it runs at most `m + 1`
    times. If a pass finds no reachable column, the visited rows have no
    allowed cell outside the visited columns -- that is exactly a violation
    of Hall's condition, and `_Infeasible` is raised with the witness.

    Infinity is only ever *compared against*, never used in arithmetic, so
    Fraction and Decimal costs work too.
    """
    u: list[Any] = [0] * n  # row potentials
    v: list[Any] = [0] * (m + 1)  # column potentials; v[m] is a virtual column
    p = [-1] * (m + 1)  # p[j] = row matched to column j (p[m] = current row)
    way = [0] * (m + 1)
    inf = _INF

    # Warm start (what Munkres steps 1-2 do): reduce each row by its cheapest
    # allowed cell, then greedily match rows to still-free zero-reduced cells.
    # Only a row reduction is sound here -- a column reduction would break
    # optimality when there are more columns than rows. A row with no allowed
    # cell at all can never be matched.
    matched = [False] * n
    for i in range(n):
        allowed = edges[i]
        if not allowed:
            raise _Infeasible([i], [])
        low = min(cost for _, cost in allowed)
        u[i] = low
        if events is not None:
            events.append({"event": "row_reduce", "row": i, "min": low})
        for j, cost in allowed:
            if p[j] == -1 and cost == low:
                p[j] = i
                matched[i] = True
                if events is not None:
                    events.append({"event": "star", "row": i, "col": j})
                break

    for i in range(n):
        if matched[i]:
            continue
        if events is not None:
            events.append({"event": "augment_start", "row": i})
        p[m] = i
        j0 = m
        minv: list[Any] = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            ui = u[i0]
            for j, cost in edges[i0]:
                if not used[j]:
                    cur = cost - ui - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
            delta: Any = inf
            j1 = -1
            for j in range(m):
                if not used[j]:
                    mv = minv[j]
                    if mv < delta:
                        delta = mv
                        j1 = j
            if j1 < 0:
                raise _Infeasible(
                    sorted(p[j] for j in range(m + 1) if used[j]),
                    [j for j in range(m) if used[j]],
                )
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    mv = minv[j]
                    if mv is not inf:
                        minv[j] = mv - delta
            if events is not None:
                held = p[j1] if p[j1] != -1 else None
                events.append(
                    {"event": "adjust", "delta": delta, "reached_col": j1, "matched_row": held}
                )
            j0 = j1
            if p[j0] == -1:
                break
        path: list[int] = []
        while j0 != m:  # flip the augmenting path
            path.append(j0)
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        if events is not None:
            events.append({"event": "augmented", "row": i, "path": path[::-1]})

    if duals is not None:
        duals.extend([u, v[:m]])
    return [(p[j], j) for j in range(m) if p[j] >= 0]


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.

    Instances hold **no state**: every call works on local data, so a single
    `Munkres` object may be shared freely between threads.
    """

    def __init__(self) -> None:
        """Create a new instance"""

    def pad_matrix(self, matrix: MatrixLike, pad_value: Number = 0) -> list[list[Any]]:
        """
        Pad a possibly non-square matrix to make it square.

        **Parameters**

        - `matrix`: matrix to pad
        - `pad_value`: value to use to pad the matrix

        **Returns**

        a new, possibly padded, matrix. The input is never modified, whatever
        its type (list, tuple, numpy array...).
        """
        rows = _as_lists(matrix)
        max_columns = max((len(row) for row in rows), default=0)
        total_rows = max(max_columns, len(rows))

        new_matrix = [row + [pad_value] * (total_rows - len(row)) for row in rows]
        while len(new_matrix) < total_rows:
            new_matrix.append([pad_value] * total_rows)
        return new_matrix

    def compute(self, cost_matrix: MatrixLike) -> list[tuple[int, int]]:
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of `(row, column)` tuples
        that can be used to traverse the matrix.

        The matrix may be square or rectangular. If it is rectangular, every
        row (or column, whichever is fewer) is matched, so `min(rows, columns)`
        pairs are returned. The caller's matrix is never modified.

        **Parameters**

        - `cost_matrix`: The cost matrix (nested lists/tuples, numpy array or
          pandas DataFrame). Cells may be `DISALLOWED` (or `+inf`) to forbid
          a pairing.

        **Returns**

        A list of `(row, column)` tuples, sorted by row, that describe the
        lowest cost assignment. An empty matrix (no rows, or no columns) has
        exactly one valid assignment, the empty one, so `[]` is returned.

        **Raises**

        - `ValueError`: the matrix is not rectangular, or contains `NaN` or
          `-inf`.
        - `TypeError`: a cell is not a number.
        - `UnsolvableMatrix`: the forbidden cells make a complete assignment
          impossible. The exception's `rows` and `cols` attributes name a
          group of rows that compete for too few columns.
        """
        n_rows, n_cols, edges = _validated_edges(cost_matrix)
        return _assign(n_rows, n_cols, edges)


def _explain(bad: _Infeasible, left: str, right: str, *, swapped: bool) -> UnsolvableMatrix:
    """Turn a Hall-violation witness into a readable UnsolvableMatrix."""
    lefts, rights = bad.rows, bad.cols
    if len(lefts) == 1 and not rights:
        message = f"{left.capitalize()} {lefts[0]} is entirely DISALLOWED."
    else:
        plural = "" if len(rights) == 1 else "s"
        message = (
            f"No complete assignment exists: {left}s {lefts} can only be "
            f"matched to {right}s {rights} ({len(lefts)} {left}s compete "
            f"for {len(rights)} {right}{plural})."
        )
    rows, cols = (rights, lefts) if swapped else (lefts, rights)
    return UnsolvableMatrix(message, rows=rows, cols=cols)


def _assign(
    n_rows: int,
    n_cols: int,
    edges: Sequence[Sequence[tuple[int, Any]]],
    *,
    gate: Any = None,
    trace: Trace | None = None,
    duals: list[Any] | None = None,
) -> list[tuple[int, int]]:
    """
    Solve a validated problem. Orientation, gating and tracing live here so
    `Munkres.compute` and `solve()` share one code path.

    Without `gate`, `min(rows, columns)` pairs are returned and
    `UnsolvableMatrix` is raised if forbidden cells make that impossible.

    With `gate` (a cost threshold), pairs costing more than `gate` are never
    made and every row may be left unmatched at a price of `gate`, so the
    result minimises ``sum(cost of matched pairs) + gate * (unmatched rows)``
    (equivalently: it maximises the total of ``gate - cost`` over the pairs
    made). This cannot be infeasible.
    """
    transposed = n_rows > n_cols
    if transposed:
        graph: list[list[tuple[int, Any]]] = [[] for _ in range(n_cols)]
        for i, allowed in enumerate(edges):
            for j, cost in allowed:
                graph[j].append((i, cost))
        n, m = n_cols, n_rows
    else:
        graph = [list(allowed) for allowed in edges]
        n, m = n_rows, n_cols

    width = m
    if gate is not None:
        graph = [
            [(j, c) for j, c in allowed if c <= gate] + [(m + k, gate)]
            for k, allowed in enumerate(graph)
        ]
        width = m + n

    events: list[dict[str, Any]] | None = None
    if trace is not None:
        trace.transposed = transposed
        events = trace.events
    try:
        pairs = _solve(n, width, graph, events, duals)
    except _Infeasible as bad:
        left, right = ("column", "row") if transposed else ("row", "column")
        raise _explain(bad, left, right, swapped=transposed) from None
    if gate is not None:
        pairs = [(r, c) for r, c in pairs if c < m]
    result = sorted((c, r) for r, c in pairs) if transposed else sorted(pairs)
    if events is not None:
        events.append({"event": "done", "pairs": result})
    return result


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def make_cost_matrix(
    profit_matrix: MatrixLike,
    inversion_function: Callable[[Any], Any] | None = None,
) -> list[list[Any]]:
    """
    Create a cost matrix from a profit matrix by calling `inversion_function()`
    to invert each value. The inversion function must take one numeric argument
    (of any type) and return another numeric argument which is presumed to be
    the cost inverse of the original profit value. If the inversion function
    is not provided, a given cell's inverted value is calculated as
    `max(matrix) - value`.

    `DISALLOWED` (and `+inf`) cells are forbidden pairings, not profits: they
    are carried over unchanged, ignored when finding `max(matrix)`, and never
    passed to the inversion function.

    This is a module-level function. Call it like this:

        from munkres import make_cost_matrix
        cost_matrix = make_cost_matrix(matrix, inversion_func)

    For example:

        from munkres import make_cost_matrix
        cost_matrix = make_cost_matrix(matrix, lambda x : 1000 - x)

    **Parameters**

    - `profit_matrix`: The matrix to convert from profit to cost values.
    - `inversion_function`: The function to use to invert each entry in the
      profit matrix.

    **Returns**

    A new matrix representing the inversion of `profit_matrix`.
    """
    rows = _as_lists(profit_matrix)

    def forbidden(value: Any) -> bool:
        return value is DISALLOWED or (isinstance(value, float) and value == _INF)

    invert: Callable[[Any], Any]
    if inversion_function is not None:
        invert = inversion_function
    else:
        # (if nothing is usable, every cell is forbidden and invert is never called)
        maximum = max((v for row in rows for v in row if not forbidden(v)), default=0)

        def invert(x: Any) -> Any:
            return maximum - x

    return [[value if forbidden(value) else invert(value) for value in row] for row in rows]


def print_matrix(matrix: MatrixLike, msg: str | None = None) -> None:
    """
    Convenience function: Displays the contents of a matrix.

    **Parameters**

    - `matrix`: The matrix to print
    - `msg`: Optional message to print before displaying the matrix
    """
    rows = _as_lists(matrix)

    if msg is not None:
        print(msg)

    def shown(val: Any) -> Any:
        return DISALLOWED_PRINTVAL if val is DISALLOWED else val

    # Calculate the appropriate format width.
    width = max((len(str(shown(val))) for row in rows for val in row), default=0)

    # Make the format string
    fmt = f"%{width}s"

    # Print the matrix
    for row in rows:
        sep = "["
        for val in row:
            sys.stdout.write(sep + fmt % (shown(val),))
            sep = ", "
        sys.stdout.write("]\n")
