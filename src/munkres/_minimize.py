# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Counterexample minimiser: shrink a failing matrix while a user predicate keeps failing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from munkres._core import DISALLOWED, _as_lists

__all__ = ["minimize_failing_matrix"]


def _rank(v: Any) -> int:
    return 0 if v is DISALLOWED else (1 if v == 0 else 2)


def minimize_failing_matrix(
    matrix: Any, fails: Callable[[list[list[Any]]], bool]
) -> list[list[Any]]:
    """
    Greedy delta-debugging: repeatedly delete a row or column, then simplify cells to 0, then DISALLOWED, keeping any change after which `fails(matrix)` is still True. `fails` must be True
    for the input. The result is 1-minimal for these moves (no single move keeps it failing).
    """
    cur = [list(r) for r in _as_lists(matrix)]
    if not fails(cur):
        raise ValueError("the predicate does not fail on the given matrix")
    changed = True
    while changed:
        changed = False
        for i in range(len(cur)):
            cand = cur[:i] + cur[i + 1 :]
            if cand and fails(cand):
                cur, changed = cand, True
                break
        if changed:
            continue
        for j in range(len(cur[0]) if cur else 0):
            cand = [r[:j] + r[j + 1 :] for r in cur]
            if cand[0] and fails(cand):
                cur, changed = cand, True
                break
        if changed:
            continue
        for i in range(len(cur)):
            for j in range(len(cur[0])):
                for repl in (0, DISALLOWED):
                    if _rank(repl) >= _rank(cur[i][j]):
                        continue  # only ever simplify (other -> 0 -> DISALLOWED), so this terminates
                    cand = [list(r) for r in cur]
                    cand[i][j] = repl
                    if fails(cand):
                        cur, changed = cand, True
                        break
                if changed:
                    break
            if changed:
                break
    return cur
