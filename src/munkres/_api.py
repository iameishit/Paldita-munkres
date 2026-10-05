# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""The 2.x high-level API: `solve`, `Assignment`, `diagnose`, `linear_sum_assignment`."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from munkres._core import (
    _INF,
    _NUMBER_TYPES,
    DISALLOWED,
    MatrixLike,
    UnsolvableMatrix,
    _as_lists,
    _assign,
    _validated_edges,
)
from munkres._trace import Trace

__all__ = [
    "Assignment",
    "Diagnosis",
    "build_cost_matrix",
    "diagnose",
    "linear_sum_assignment",
    "solve",
]

_A = TypeVar("_A")
_B = TypeVar("_B")
_NO_PAIRS: Any = object()  # sentinel: a threshold so strict that nothing can be matched


@dataclass(frozen=True)
class Assignment:
    """
    The result of `solve`.

    - `pairs`: the matched `(row, column)` index pairs, sorted by row
    - `total`: the sum of *your original* matrix values over `pairs`
      (the total cost, or the total profit when `maximize=True`)
    - `unmatched_rows` / `unmatched_cols`: indexes left without a partner
      (always the surplus side of a rectangular matrix, plus anything gated out)
    - `shape`: `(rows, columns)` of the input
    - `row_labels` / `col_labels`: the DataFrame's index / columns, if you passed one
    - `trace`: a `Trace` if you asked for one

    Iterating an `Assignment` yields its pairs, so it can stand in for the list
    `Munkres().compute()` returns. Note that an empty one is falsy.
    """

    pairs: tuple[tuple[int, int], ...]
    total: Any
    unmatched_rows: tuple[int, ...]
    unmatched_cols: tuple[int, ...]
    shape: tuple[int, int]
    maximize: bool = False
    row_labels: tuple[Any, ...] | None = None
    col_labels: tuple[Any, ...] | None = None
    trace: Trace | None = None

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def rows(self) -> tuple[int, ...]:
        """Matched row indexes, in order (like SciPy's `row_ind`)."""
        return tuple(r for r, _ in self.pairs)

    @property
    def cols(self) -> tuple[int, ...]:
        """Matched column indexes, aligned with `rows` (like SciPy's `col_ind`)."""
        return tuple(c for _, c in self.pairs)

    def as_dict(self) -> dict[int, int]:
        """`{row: column}` for every matched row."""
        return dict(self.pairs)

    def labelled(self) -> list[tuple[Any, Any]]:
        """The pairs as `(row label, column label)`; plain indexes if there are no labels."""
        return [
            (
                self.row_labels[r] if self.row_labels is not None else r,
                self.col_labels[c] if self.col_labels is not None else c,
            )
            for r, c in self.pairs
        ]


@dataclass(frozen=True)
class Diagnosis:
    """Why a matrix has no complete assignment (a violation of Hall's condition)."""

    message: str
    rows: tuple[int, ...]
    cols: tuple[int, ...]


def _negated(rows: list[list[Any]]) -> list[list[Any]]:
    """Profit -> cost by negation (exact for int, Fraction and Decimal)."""
    out: list[list[Any]] = []
    for i, row in enumerate(rows):
        new: list[Any] = []
        for j, value in enumerate(row):
            if value is DISALLOWED or not isinstance(value, _NUMBER_TYPES):
                new.append(value)  # forbidden, or a TypeError for validation to report
                continue
            try:
                infinite_profit = value == _INF
            except ArithmeticError:  # Decimal('sNaN'): validation reports it
                new.append(value)
                continue
            if infinite_profit:
                raise ValueError(
                    f"cell [{i}][{j}] is +infinity, which has no defined assignment value "
                    "(use -inf or DISALLOWED to forbid a pairing)"
                )
            new.append(-value)
        out.append(new)
    return out


def _gate(threshold: Any, *, maximize: bool) -> Any:
    """Turn the user's threshold into a cost-space gate (None = no gating)."""
    if threshold is None:
        return None
    name = "min_profit" if maximize else "max_cost"
    if not isinstance(threshold, _NUMBER_TYPES):
        raise TypeError(f"{name} must be a number, got {threshold!r}")
    if threshold != threshold:  # noqa: PLR0124
        raise ValueError(f"{name} is NaN")
    gate = -threshold if maximize else threshold
    if gate == _INF:
        return None
    if gate == -_INF:
        return _NO_PAIRS
    return gate


def _labels(matrix: Any) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    index: Any = getattr(matrix, "index", None)
    columns: Any = getattr(matrix, "columns", None)
    if hasattr(index, "tolist") and hasattr(columns, "tolist"):  # a DataFrame, not a list
        return tuple(index.tolist()), tuple(columns.tolist())
    return None, None


def solve(
    matrix: MatrixLike,
    *,
    maximize: bool = False,
    max_cost: Any = None,
    min_profit: Any = None,
    trace: bool = False,
) -> Assignment:
    """
    Solve an assignment problem and describe the answer.

    **Parameters**

    - `matrix`: costs (or profits, with `maximize=True`) as nested sequences, a
      numpy array or a pandas DataFrame. `DISALLOWED` and `+inf` (`-inf` when
      maximising) forbid a pairing.
    - `maximize`: maximise the total instead of minimising it. Exact for int,
      `Fraction` and `Decimal`, because values are negated rather than subtracted
      from a maximum.
    - `max_cost`: **gating** (minimising only). Pairs costing more than this are
      never made, and a row may be left unmatched instead. The result minimises
      the cost of the pairs made plus `max_cost` for every row left unmatched, so
      a pair is only worth making if it costs less than `max_cost`. A pair
      costing exactly `max_cost` is a tie with leaving the row unmatched. With
      gating a matrix can never be unsolvable.
    - `min_profit`: the same gate when `maximize=True`: pairs earning less than
      this are never made.
    - `trace`: record the solver's steps in `Assignment.trace`.

    **Raises**

    - `UnsolvableMatrix`: forbidden cells make a complete assignment impossible
      (not possible with gating)
    - `ValueError`: ragged matrix, `NaN`, an infinite value with no meaning, a
      threshold that does not fit the mode
    - `TypeError`: a non-numeric cell or threshold
    """
    if maximize and max_cost is not None:
        raise ValueError("max_cost is for minimising; use min_profit with maximize=True")
    if not maximize and min_profit is not None:
        raise ValueError("min_profit needs maximize=True; use max_cost when minimising")
    gate = _gate(min_profit if maximize else max_cost, maximize=maximize)

    row_labels, col_labels = _labels(matrix)
    original = _as_lists(matrix)
    n_rows, n_cols, edges = _validated_edges(_negated(original) if maximize else original)

    recorder = Trace(shape=(n_rows, n_cols)) if trace else None
    if gate is _NO_PAIRS:
        pairs: list[tuple[int, int]] = []
        if recorder is not None:
            recorder.events.append({"event": "done", "pairs": []})
    else:
        pairs = _assign(n_rows, n_cols, edges, gate=gate, trace=recorder)

    return _make_assignment(
        original, pairs, (n_rows, n_cols), maximize=maximize,
        labels=(row_labels, col_labels), trace=recorder,
    )  # fmt: skip


def _make_assignment(
    original: list[list[Any]],
    pairs: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    maximize: bool,
    labels: tuple[tuple[Any, ...] | None, tuple[Any, ...] | None] = (None, None),
    trace: Trace | None = None,
) -> Assignment:
    n_rows, n_cols = shape
    matched_rows = {r for r, _ in pairs}
    matched_cols = {c for _, c in pairs}
    return Assignment(
        pairs=tuple(pairs),
        total=sum(original[r][c] for r, c in pairs),
        unmatched_rows=tuple(i for i in range(n_rows) if i not in matched_rows),
        unmatched_cols=tuple(j for j in range(n_cols) if j not in matched_cols),
        shape=shape,
        maximize=maximize,
        row_labels=labels[0],
        col_labels=labels[1],
        trace=trace,
    )


def diagnose(matrix: MatrixLike, *, maximize: bool = False) -> Diagnosis | None:
    """
    Explain why a matrix cannot be solved, without raising.

    Returns `None` if a complete assignment exists. Otherwise returns a
    `Diagnosis` naming rows that can only use too few columns (or the reverse for
    a tall matrix), which is exactly Hall's marriage-theorem violation.
    """
    original = _as_lists(matrix)
    n_rows, n_cols, edges = _validated_edges(_negated(original) if maximize else original)
    try:
        _assign(n_rows, n_cols, edges)
    except UnsolvableMatrix as bad:
        return Diagnosis(str(bad), bad.rows, bad.cols)
    return None


def linear_sum_assignment(cost_matrix: MatrixLike, maximize: bool = False) -> tuple[Any, Any]:
    """
    A drop-in for `scipy.optimize.linear_sum_assignment`: returns
    `(row_ind, col_ind)`, sorted by row.

    The return type follows the input: numpy arrays (and DataFrames) get numpy
    index arrays back; anything else gets lists. As in SciPy, `+inf` forbids a
    pairing, an infeasible matrix raises `ValueError` (here the subclass
    `UnsolvableMatrix`), and when several optimal answers exist the one returned
    may differ from SciPy's (the total cost is always the same).
    """
    try:
        result = solve(cost_matrix, maximize=maximize)
    except TypeError as bad:
        raise ValueError(str(bad)) from None
    rows, cols = list(result.rows), list(result.cols)
    if hasattr(cost_matrix, "tolist") or hasattr(cost_matrix, "to_numpy"):
        try:
            np = importlib.import_module("numpy")
        except ImportError:
            return rows, cols
        return np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp)
    return rows, cols


def build_cost_matrix(
    rows: Iterable[_A], cols: Iterable[_B], cost: Callable[[_A, _B], Any]
) -> list[list[Any]]:
    """
    Build a matrix by calling `cost(row_item, col_item)` for every pair, e.g. a
    distance between points. If `cost` returns `None` that pairing is
    `DISALLOWED`.
    """
    col_items = list(cols)
    matrix: list[list[Any]] = []
    for a in rows:
        row: list[Any] = []
        for b in col_items:
            value = cost(a, b)
            row.append(DISALLOWED if value is None else value)
        matrix.append(row)
    return matrix
