# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Dynamic assignment: keep primal+dual state and repair it after changes.

Exact rational arithmetic, so the repaired result is *provably* optimal whenever the
invariants below hold (and `verify()` re-proves it independently with `_verify`).

Invariants after `solve()`: u_i + v_j <= c_ij for all allowed cells; matched cells are
tight; v_j == 0 on unmatched columns; every row matched. Rows <= columns is required.

Repair procedure (cf. Mills-Tettey, Stentz, Dias 2007, "Dynamic Hungarian"):
1. lower u_i to restore dual feasibility after cost decreases / new rows;
2. free columns get v_j = 0 (which may lower u of adjacent rows);
3. unmatch rows whose matched cell is no longer tight;
4. re-augment only the unmatched rows with Dijkstra on reduced costs.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, UnsolvableMatrix, _is_forbidden

__all__ = ["AssignmentState", "Delta", "RepairStats"]


@dataclass
class RepairStats:
    reaugmented_rows: int = 0
    unmatched_by_repair: int = 0
    full_resolve: bool = False
    history: list[int] = field(default_factory=list)


def _q(x: Any) -> Fraction | None:
    if x is None or _is_forbidden(x):
        return None
    if x != x:  # noqa: PLR0124
        raise ValueError("NaN cost")
    return Fraction(x)


class AssignmentState:
    """Mutable, repairable optimal assignment keyed by arbitrary hashable row/column ids."""

    def __init__(self, cost: Any = None, maximize: bool = False) -> None:
        self.maximize = maximize
        self._c: dict[Hashable, dict[Hashable, Fraction]] = {}
        self._cols: set[Hashable] = set()
        self._u: dict[Hashable, Fraction] = {}
        self._v: dict[Hashable, Fraction] = {}
        self._mr: dict[Hashable, Hashable] = {}
        self._mc: dict[Hashable, Hashable] = {}
        self.stats = RepairStats()
        self._dirty = False
        self._ops: list[tuple[Any, ...]] = []
        self._init: list[list[Any]] | None = None
        # Column creation rank: the deterministic tie-break among equally good columns (never reused).
        self._rank: dict[Hashable, int] = {}
        self._next_rank = 0
        if cost is not None:
            from munkres._core import _as_lists

            rows = _as_lists(cost)
            self._init = [list(r) for r in rows]
            for j in range(len(rows[0]) if rows else 0):
                self._cols.add(j)
                self._v[j] = Fraction(0)
                self._rank[j] = self._next_rank
                self._next_rank += 1
            for i, row in enumerate(rows):
                self._c[i] = {j: self._in(x) for j, x in enumerate(row) if _q(x) is not None}
                self._u[i] = None  # type: ignore[assignment]
            self._dirty = True

    # ----- input helpers
    def _in(self, x: Any) -> Fraction:
        q = _q(x)
        return -q if (self.maximize and q is not None) else q  # type: ignore[return-value]

    # ----- mutations
    # Every mutator is transactional: it validates and converts *all* of its input first and only then
    # touches state or the operation log, so a rejected call leaves the object exactly as it was.
    def _commit_cell(self, row: Hashable, col: Hashable, raw: Any, q: Fraction | None) -> None:
        self._ops.append(("update_cost", row, col, raw))
        if q is None:
            self._c[row].pop(col, None)
        else:
            self._c[row][col] = q
        self._dirty = True

    def update_cost(self, row: Hashable, col: Hashable, cost: Any) -> None:
        """Set a cell; `DISALLOWED`/inf/None forbids it. Row and column must exist."""
        if row not in self._c or col not in self._cols:
            raise KeyError((row, col))
        q = self._in(cost)  # may raise before anything is mutated
        self._commit_cell(row, col, cost, q)

    def update_costs(self, changes: dict[tuple[Hashable, Hashable], Any]) -> None:
        """Apply many cell updates all-or-nothing: if any entry is invalid none is applied."""
        staged = []
        for (r, c), x in changes.items():
            if r not in self._c or c not in self._cols:
                raise KeyError((r, c))
            staged.append((r, c, x, self._in(x)))
        for r, c, x, q in staged:
            self._commit_cell(r, c, x, q)

    def add_row(self, row: Hashable, costs: dict[Hashable, Any]) -> None:
        if row in self._c:
            raise KeyError(f"row {row!r} exists")
        staged: dict[Hashable, Fraction] = {}
        for c, x in costs.items():
            if c not in self._cols:
                raise KeyError(f"unknown column {c!r}")
            q = self._in(x)
            if q is not None:
                staged[c] = q
        self._ops.append(("add_row", row, dict(costs)))
        self._c[row] = staged
        self._u[row] = None  # type: ignore[assignment]
        self._dirty = True

    def add_column(self, col: Hashable, costs: dict[Hashable, Any] | None = None) -> None:
        if col in self._cols:
            raise KeyError(f"column {col!r} exists")
        staged: dict[Hashable, Fraction] = {}
        for r, x in (costs or {}).items():
            if r not in self._c:
                raise KeyError(f"unknown row {r!r}")
            q = self._in(x)
            if q is not None:
                staged[r] = q
        self._ops.append(("add_column", col, dict(costs or {})))
        self._cols.add(col)
        self._v[col] = Fraction(0)
        self._rank[col] = self._next_rank
        self._next_rank += 1
        for r, q in staged.items():
            self._c[r][col] = q
        self._dirty = True

    def remove_row(self, row: Hashable) -> None:
        if row not in self._c:
            raise KeyError(f"unknown row {row!r}")
        self._ops.append(("remove_row", row))
        self._unmatch_row(row)
        del self._c[row]
        self._u.pop(row, None)
        self._dirty = True

    def remove_column(self, col: Hashable) -> None:
        if col not in self._cols:
            raise KeyError(f"unknown column {col!r}")
        self._ops.append(("remove_column", col))
        if col in self._mc:
            self._unmatch_row(self._mc[col])
        self._cols.discard(col)
        self._v.pop(col, None)
        self._rank.pop(col, None)
        for r in self._c.values():
            r.pop(col, None)
        self._dirty = True

    def _unmatch_row(self, row: Hashable) -> None:
        col = self._mr.pop(row, None)
        if col is not None:
            self._mc.pop(col, None)

    # ----- repair + solve
    def _repair_duals(self) -> bool:
        """Restore invariants 1-3. Returns False if it failed to converge (caller resolves)."""
        limit = 20 * (len(self._c) + len(self._cols)) + 100
        for _ in range(limit):
            changed = False
            for col in self._cols:
                if col not in self._mc and self._v[col] != 0:
                    self._v[col] = Fraction(0)
                    changed = True
            for r, row in self._c.items():
                if not row:
                    self._u[r] = None  # type: ignore[assignment]
                    continue
                best = min(c - self._v[j] for j, c in row.items())
                if self._u[r] is None or self._u[r] > best:
                    self._u[r] = best
                    changed = True
            for r in list(self._mr):
                j = self._mr[r]
                c = self._c[r].get(j)
                if c is None or self._u[r] is None or c - self._u[r] - self._v[j] != 0:
                    self._unmatch_row(r)
                    self.stats.unmatched_by_repair += 1
                    changed = True
            if not changed:
                for r, row in self._c.items():  # rows never seen get their maximal feasible dual
                    if row and self._u[r] is None:
                        self._u[r] = min(c - self._v[j] for j, c in row.items())
                return True
        return False

    def _augment(self, r0: Hashable) -> None:
        u, v, c = self._u, self._v, self._c
        dist: dict[Hashable, Fraction] = {}
        best: dict[Hashable, tuple[Fraction, Hashable]] = {}
        prev: dict[Hashable, Hashable] = {}
        rows_in = {r0: Fraction(0)}

        def relax(i: Hashable, base: Fraction) -> None:
            for j, cij in c[i].items():
                if j in dist:
                    continue
                d = base + cij - u[i] - v[j]
                if j not in best or d < best[j][0]:
                    best[j] = (d, i)

        relax(r0, Fraction(0))
        sink = None
        while True:
            cand = [(d, j) for j, (d, _) in best.items() if j not in dist]
            if not cand:
                raise UnsolvableMatrix(f"row {r0!r} cannot be matched (Hall violation)")
            dmin = min(d for d, _ in cand)
            # Tie-break (deterministic, independent of dict traversal order): among the columns at the
            # minimum distance take a free column first (it ends the search), then the column created
            # earliest. Ties therefore resolve the same way for the same history of operations.
            j = min((j for d, j in cand if d == dmin), key=lambda c: (c in self._mc, self._rank[c]))
            dist[j] = dmin
            prev[j] = best[j][1]
            if j not in self._mc:
                sink = j
                break
            i2 = self._mc[j]
            rows_in[i2] = dmin
            relax(i2, dmin)
        big_d = dist[sink]
        u[r0] += big_d
        for j, dj in dist.items():
            if j == sink:
                continue
            u[self._mc[j]] += big_d - dj
            v[j] -= big_d - dj
        j = sink
        while True:
            i = prev[j]
            nxt = self._mr.get(i)
            self._mr[i], self._mc[j] = j, i
            if i == r0:
                break
            j = nxt
        self.stats.reaugmented_rows += 1

    def _full_reset(self) -> None:
        self._mr.clear()
        self._mc.clear()
        for j in self._cols:
            self._v[j] = Fraction(0)
        for r in self._c:
            self._u[r] = None  # type: ignore[assignment]
        self.stats.full_resolve = True

    def solve(self) -> dict[Hashable, Hashable]:
        """Repair to optimality; returns `{row: col}`. Raises `UnsolvableMatrix` if infeasible."""
        if len(self._c) > len(self._cols):
            raise UnsolvableMatrix("more rows than columns")
        if self._dirty:
            self.stats = RepairStats()  # stats describe the most recent actual repair
            if not self._repair_duals():
                self._full_reset()
                self._repair_duals()
            for r in list(self._c):
                if r not in self._mr:
                    if not self._c[r]:
                        raise UnsolvableMatrix(f"row {r!r} has no allowed column")
                    self._augment(r)
            self.stats.history.append(self.stats.reaugmented_rows)
            self._dirty = False
        return dict(self._mr)

    # ----- event log / replay
    def export_log(self) -> dict[str, Any]:
        """Everything needed to rebuild this state: the initial matrix, sense and every mutation in order."""
        return {"init": self._init, "maximize": self.maximize, "ops": list(self._ops)}

    @classmethod
    def from_log(cls, log: dict[str, Any]) -> AssignmentState:
        """Rebuild a state by replaying `export_log()`; the result is solved and verifiable like the original."""
        st = cls(log["init"], maximize=log["maximize"])
        for op in log["ops"]:
            getattr(st, op[0])(*op[1:])
        return st

    def state_at(self, step: int) -> AssignmentState:
        """The state after the first `step` logged mutations (time travel for the dynamic state)."""
        log = self.export_log()
        return AssignmentState.from_log({**log, "ops": log["ops"][:step]})

    def rewind(self, k: int = 1) -> None:
        """Undo the last `k` mutations by rebuilding from the log."""
        prior = self.state_at(max(0, len(self._ops) - k))
        self._c, self._cols, self._u, self._v = prior._c, prior._cols, prior._u, prior._v
        self._mr, self._mc, self._dirty, self._ops = prior._mr, prior._mc, prior._dirty, prior._ops
        self._rank, self._next_rank = prior._rank, prior._next_rank

    def apply(self, delta: Delta) -> None:
        """Apply a `Delta` of cell updates."""
        self.update_costs(delta.changes)

    # ----- snapshots
    def snapshot(self) -> AssignmentState:
        """Independent copy of the full primal/dual state (exact numbers are immutable, so this is cheap)."""
        out = AssignmentState(maximize=self.maximize)
        out._c = {r: dict(row) for r, row in self._c.items()}
        out._cols, out._u, out._v = set(self._cols), dict(self._u), dict(self._v)
        out._mr, out._mc, out._dirty = dict(self._mr), dict(self._mc), self._dirty
        out._rank, out._next_rank = dict(self._rank), self._next_rank
        out._ops, out._init = list(self._ops), self._init
        return out

    def restore(self, snap: AssignmentState) -> None:
        """Roll this state back to `snap` (a value from `snapshot()`); later changes to `snap` do not affect it."""
        copy = snap.snapshot()
        self.maximize = copy.maximize
        self._c, self._cols, self._u, self._v = copy._c, copy._cols, copy._u, copy._v
        self._mr, self._mc, self._dirty = copy._mr, copy._mc, copy._dirty
        self._rank, self._next_rank = copy._rank, copy._next_rank
        self._ops, self._init = copy._ops, copy._init  # the log must describe the restored state

    # ----- results
    @property
    def pairs(self) -> list[tuple[Hashable, Hashable]]:
        self.solve()
        return sorted(self._mr.items(), key=lambda p: (str(p[0]), str(p[1])))

    @property
    def objective(self) -> Fraction:
        self.solve()
        tot = sum((self._c[r][c] for r, c in self._mr.items()), Fraction(0))
        return -tot if self.maximize else tot

    def dense(self) -> tuple[list[Hashable], list[Hashable], list[list[Any]]]:
        rs = sorted(self._c, key=str)
        cs = sorted(self._cols, key=str)
        sign = -1 if self.maximize else 1
        return (
            rs,
            cs,
            [[sign * self._c[r][c] if c in self._c[r] else DISALLOWED for c in cs] for r in rs],
        )

    def verify(self) -> Any:
        """Independent exact proof that the current assignment is optimal (see `_verify`)."""
        from munkres._verify import verify_pairs

        self.solve()
        rs, cs, mat = self.dense()
        ri, ci = {r: k for k, r in enumerate(rs)}, {c: k for k, c in enumerate(cs)}
        return verify_pairs(mat, [(ri[r], ci[c]) for r, c in self._mr.items()], self.maximize)


@dataclass(frozen=True)
class Delta:
    """A set of cell updates `{(row, col): cost}`; `compose` gives the single delta equal to applying both in order."""

    changes: dict[tuple[Hashable, Hashable], Any] = field(default_factory=dict)

    def compose(self, later: Delta) -> Delta:
        return Delta({**self.changes, **later.changes})
