# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Parametric assignment: the optimum of  C(lambda) = A + lambda * B  as lambda moves over [lo, hi].
f(lambda) = min over assignments of (a_M + lambda * b_M) is piecewise linear and concave; the exact
breakpoints are found with the Eisner-Severance recursion (one LAP per probe, rational arithmetic).
The number of breakpoints can be large in the worst case, so a `max_pieces` guard is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists

__all__ = ["Piece", "optimality_interval", "parametric_assignment"]


@dataclass(frozen=True)
class Piece:
    """One linear piece of the parametric optimum envelope."""

    start: Fraction
    end: Fraction
    pairs: tuple[tuple[int, int], ...]
    base: Fraction  # sum of A over pairs
    slope: Fraction  # sum of B over pairs  (objective = base + slope * lambda)


def _line(a: list[list[Any]], b: list[list[Any]], pairs: Any) -> tuple[Fraction, Fraction]:
    return sum((Fraction(a[i][j]) for i, j in pairs), Fraction(0)), sum(
        (Fraction(b[i][j]) for i, j in pairs), Fraction(0)
    )


def parametric_assignment(
    a: Any, b: Any, lo: Any, hi: Any, maximize: bool = False, max_pieces: int = 10_000
) -> list[Piece]:
    """Pieces (in order) of the exact optimum envelope on [lo, hi]; cells forbidden in `a` stay forbidden."""
    from munkres._api import solve

    A, B = _as_lists(a), _as_lists(b)
    lo, hi = Fraction(lo), Fraction(hi)
    if lo > hi:
        raise ValueError("lo must be <= hi")
    sign = -1 if maximize else 1

    def probe(lam: Fraction) -> tuple[tuple[tuple[int, int], ...], Fraction, Fraction]:
        mat = [
            [
                DISALLOWED
                if A[i][j] is DISALLOWED
                else sign * (Fraction(A[i][j]) + lam * Fraction(B[i][j]))
                for j in range(len(A[0]))
            ]
            for i in range(len(A))
        ]
        pairs = tuple(sorted(solve(mat).pairs))
        base, slope = _line(A, B, pairs)
        return pairs, base, slope

    p_lo, p_hi = probe(lo), probe(hi)
    cuts: list[
        tuple[Fraction, tuple[Any, Fraction, Fraction]]
    ] = []  # (lambda, line active just right of it)

    def rec(l1: Fraction, m1: Any, l2: Fraction, m2: Any) -> None:
        if len(cuts) > max_pieces:
            raise RuntimeError(f"more than {max_pieces} pieces")
        (_, a1, b1), (_, a2, b2) = m1, m2
        if (a1, b1) == (a2, b2) or b1 == b2:
            return
        lam = (a2 - a1) / (b1 - b2)
        lam = min(max(lam, l1), l2)
        probe_m = probe(lam)
        val = sign * (probe_m[1] + lam * probe_m[2])
        if val == sign * (a1 + lam * b1):  # envelope has a kink exactly at lam
            cuts.append((lam, m2))
            return
        rec(l1, m1, lam, probe_m)
        rec(lam, probe_m, l2, m2)

    rec(lo, p_lo, hi, p_hi)
    cuts.sort(key=lambda t: t[0])
    pieces: list[Piece] = []
    start, cur = lo, p_lo
    for lam, nxt in cuts:
        if lam > start or (not pieces and lam == start == lo):
            pieces.append(Piece(start, lam, cur[0], cur[1], cur[2]))
        start, cur = lam, nxt
    pieces.append(Piece(start, hi, cur[0], cur[1], cur[2]))
    merged: list[Piece] = []
    for (
        p
    ) in pieces:  # merge pieces on the same line, drop zero-width pieces unless it is the only one
        if merged and (merged[-1].base, merged[-1].slope) == (p.base, p.slope):
            merged[-1] = Piece(merged[-1].start, p.end, merged[-1].pairs, p.base, p.slope)
        elif p.end > p.start or (not merged and p.start == p.end):
            merged.append(p)
    return merged or [Piece(lo, hi, p_lo[0], p_lo[1], p_lo[2])]


def optimality_interval(
    a: Any, b: Any, pairs: Any, lo: Any, hi: Any, maximize: bool = False
) -> tuple[Fraction, Fraction] | None:
    """The sub-interval of [lo, hi] on which `pairs` is optimal for A + lambda B (None if it never is)."""
    A, B = _as_lists(a), _as_lists(b)
    line = _line(A, B, pairs)
    hits = [p for p in parametric_assignment(a, b, lo, hi, maximize) if (p.base, p.slope) == line]
    if not hits:
        return None
    return hits[0].start, hits[-1].end
