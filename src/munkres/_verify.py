# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Exact optimality certificates and an independent verifier.

Design rule: the verifier shares **no code** with the solvers. `certify()` does
not trust any solver output except the proposed pairs; it re-derives exact dual
potentials by Bellman-Ford shortest paths on the residual graph (in `Fraction`
arithmetic, so float costs are treated as the exact rationals they are). A
negative residual cycle is a proof that the pairs are *not* optimal.
`Certificate.verify()` then re-checks LP duality from the stored duals alone.

Problem solved (n rows, m columns, after orienting so n <= m):
    minimise sum c_ij x_ij   s.t. each row used exactly once, each column at most once.
Dual:  maximise sum u_i + sum v_j  s.t.  u_i + v_j <= c_ij (allowed cells),  v_j <= 0.
Optimality <=> primal feasible, dual feasible, tight on matched cells, v_j = 0 on
unmatched columns, equal objectives.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

from munkres._core import _as_lists, _is_forbidden, _validated_edges

__all__ = [
    "Certificate",
    "CertificateCache",
    "Verification",
    "certify",
    "problem_hash",
    "verify_pairs",
]

CERT_VERSION = 1


def _digits(n: int) -> str:
    """Decimal text of `n` (unchanged for normal sizes); hex with a marker beyond Python's int->str digit limit."""
    try:
        return str(n)
    except ValueError:  # sys.get_int_max_str_digits() exceeded
        return "h" + format(n, "x")


def _frac(x: Any) -> Fraction:
    if isinstance(x, Decimal):
        return Fraction(x)
    return Fraction(x)


def _exact_rows(matrix: Any, maximize: bool) -> tuple[int, int, list[dict[int, Fraction]]]:
    """Exact minimisation-form costs: rows[i][j] = cost (negated if maximize), allowed cells only."""
    original = _as_lists(matrix)
    if maximize:
        from munkres._api import _negated

        original = _negated(original)
    n, m, edges = _validated_edges(original)
    return n, m, [{j: _frac(c) for j, c in allowed} for allowed in edges]


def problem_hash(matrix: Any, maximize: bool = False) -> str:
    """Canonical SHA-256 of the problem (shape, sense, exact cell values, forbidden pattern)."""
    original = _as_lists(matrix)
    h = hashlib.sha256()
    h.update(
        f"paldita-problem-v1|{len(original)}|{len(original[0]) if original else 0}|{int(maximize)}".encode()
    )
    for row in original:
        for v in row:
            if _is_forbidden(v):
                h.update(b"|X")
            else:
                f = _frac(v)
                h.update(f"|{_digits(f.numerator)}/{_digits(f.denominator)}".encode())
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class Verification:
    """Outcome of a check: `ok` plus a precise `reason` and the failed `stage`."""

    ok: bool
    stage: str
    reason: str
    objective: Fraction | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class Certificate:
    """Portable proof that `pairs` is an optimal assignment of the problem with `problem_hash`."""

    version: int
    problem_hash: str
    shape: tuple[int, int]
    maximize: bool
    pairs: tuple[tuple[int, int], ...]
    objective: Fraction  # in the caller's sense (profit if maximize)
    u: tuple[Fraction, ...]  # duals for the oriented problem (rows <= cols)
    v: tuple[Fraction, ...]
    transposed: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def verify(self, matrix: Any) -> Verification:
        """Independently re-check this certificate against `matrix` (exact arithmetic)."""
        if problem_hash(matrix, self.maximize) != self.problem_hash:
            return Verification(False, "hash", "certificate was issued for a different problem")
        n, m, rows = _exact_rows(matrix, self.maximize)
        if (n, m) != self.shape:
            return Verification(False, "shape", f"shape {(n, m)} != certified {self.shape}")
        return _check_duals(
            n, m, rows, self.pairs, self.u, self.v, self.transposed, self.maximize, self.objective
        )

    def to_json(self) -> str:
        d = {
            "version": self.version,
            "problem_hash": self.problem_hash,
            "shape": list(self.shape),
            "maximize": self.maximize,
            "pairs": [list(p) for p in self.pairs],
            "objective": str(self.objective),
            "u": [str(x) for x in self.u],
            "v": [str(x) for x in self.v],
            "transposed": self.transposed,
            "meta": self.meta,
        }
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Certificate:
        """Parse a certificate. Any malformed or hostile input raises `ValueError` (never another exception type)."""
        try:
            d = json.loads(text)
            if not isinstance(d, dict):
                raise ValueError("certificate must be a JSON object")
            if d.get("version") != CERT_VERSION:
                raise ValueError(f"unsupported certificate version {d.get('version')!r}")
            shape = d["shape"]
            if not (
                isinstance(shape, list)
                and len(shape) == 2
                and all(type(x) is int and x >= 0 for x in shape)
            ):
                raise ValueError("shape must be two non-negative integers")
            pairs = d["pairs"]
            if not (
                isinstance(pairs, list)
                and all(
                    isinstance(p, list) and len(p) == 2 and all(type(x) is int for x in p)
                    for p in pairs
                )
            ):
                raise ValueError("pairs must be a list of integer pairs")
            for key in ("u", "v"):
                if not (isinstance(d[key], list) and all(isinstance(x, str) for x in d[key])):
                    raise ValueError(f"{key} must be a list of rational strings")
            if not (isinstance(d["problem_hash"], str) and isinstance(d["objective"], str)):
                raise ValueError("problem_hash and objective must be strings")
            if type(d["maximize"]) is not bool or type(d["transposed"]) is not bool:
                raise ValueError("maximize and transposed must be booleans")
            meta = d.get("meta", {})
            if not isinstance(meta, dict):
                raise ValueError("meta must be an object")
            return cls(
                CERT_VERSION,
                d["problem_hash"],
                (shape[0], shape[1]),
                d["maximize"],
                tuple((a, b) for a, b in pairs),
                Fraction(d["objective"]),
                tuple(Fraction(x) for x in d["u"]),
                tuple(Fraction(x) for x in d["v"]),
                d["transposed"],
                dict(meta),
            )
        except ValueError:
            raise
        except (
            KeyError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            AttributeError,
            RecursionError,
        ) as bad:
            raise ValueError(f"malformed certificate: {type(bad).__name__}: {bad}") from None


def _orient(
    n: int, m: int, rows: list[dict[int, Fraction]]
) -> tuple[int, int, list[dict[int, Fraction]], bool]:
    if n <= m:
        return n, m, rows, False
    t: list[dict[int, Fraction]] = [{} for _ in range(m)]
    for i, r in enumerate(rows):
        for j, c in r.items():
            t[j][i] = c
    return m, n, t, True


def _check_duals(  # noqa: PLR0911
    n: int,
    m: int,
    rows: Any,
    pairs: Any,
    u: Any,
    v: Any,
    transposed: bool,
    maximize: bool,
    objective: Any,
) -> Verification:
    """The small, boring checker: LP duality on exact numbers."""
    on, om, orows, tr = _orient(n, m, rows)
    if tr != transposed:
        return Verification(False, "orientation", "orientation mismatch")
    if len(u) != on or len(v) != om:
        return Verification(False, "dual-shape", "dual vector lengths do not match the problem")
    mp = [(c, r) if transposed else (r, c) for r, c in pairs]  # oriented (row, col)
    if len(mp) != on or len({r for r, _ in mp}) != on or len({c for _, c in mp}) != len(mp):
        return Verification(False, "primal", f"need {on} distinct rows matched to distinct columns")
    total = Fraction(0)
    for r, c in mp:
        if not (0 <= r < on and 0 <= c < om) or c not in orows[r]:
            return Verification(False, "primal", f"pair {(r, c)} is out of range or forbidden")
        total += orows[r][c]
        if u[r] + v[c] != orows[r][c]:
            return Verification(False, "slackness", f"matched cell {(r, c)} is not tight")
    for i in range(on):
        for j, c in orows[i].items():
            if u[i] + v[j] > c:
                return Verification(False, "dual", f"dual infeasible at {(i, j)}")
    used = {c for _, c in mp}
    for j in range(om):
        if v[j] > 0 or (j not in used and v[j] != 0):
            return Verification(False, "dual", f"column {j} violates v<=0 / v=0 on unmatched")
    if sum(u) + sum(v) != total:
        return Verification(False, "duality-gap", "dual objective differs from primal objective")
    shown = -total if maximize else total
    if objective is not None and shown != objective:
        return Verification(
            False, "objective", "stated objective differs from recomputed objective"
        )
    return Verification(
        True, "ok", "optimal: primal feasible, dual feasible, zero duality gap", shown
    )


def _padded_duals(on: int, om: int, orows: Any, mp: Any) -> Any:
    """
    Exact duals of the zero-padded square problem for the given matching, via
    Bellman-Ford on the residual graph. Returns `(u, v, match, size, full)` or a
    failed `Verification` carrying a negative-cycle witness.
    """
    free = [j for j in range(om) if j not in {c for _, c in mp}]
    size = om
    full = [dict(r) for r in orows] + [{j: Fraction(0) for j in range(om)} for _ in range(om - on)]
    match = dict(mp)
    for k, j in enumerate(free):
        match[on + k] = j
    adj: list[list[tuple[int, Fraction]]] = [[] for _ in range(2 * size)]
    for i in range(size):
        for j, c in full[i].items():
            adj[i].append((size + j, c))  # every allowed cell keeps its forward arc ...
            if match[i] == j:
                adj[size + j].append(
                    (i, -c)
                )  # ... a matched cell also gets the reverse arc (=> tight)
    dist = [Fraction(0)] * (2 * size)
    pred = [-1] * (2 * size)
    count = [0] * (2 * size)
    inq = [True] * (2 * size)
    q = deque(range(2 * size))
    while q:
        x = q.popleft()
        inq[x] = False
        for y, w in adj[x]:
            if dist[x] + w < dist[y]:
                dist[y] = dist[x] + w
                pred[y] = x
                count[y] += 1
                if count[y] > 2 * size:  # negative cycle
                    z = pred[y]
                    for _ in range(2 * size):
                        z = pred[z]
                    cyc = [z]
                    w2 = pred[z]
                    while w2 != z:
                        cyc.append(w2)
                        w2 = pred[w2]
                    shown = [("row", t) if t < size else ("col", t - size) for t in reversed(cyc)]
                    return Verification(
                        False, "optimality", f"suboptimal: negative alternating cycle {shown}"
                    )
                if not inq[y]:
                    inq[y] = True
                    q.append(y)
    return [-dist[i] for i in range(size)], [dist[size + j] for j in range(size)], match, size, full


def certify(matrix: Any, pairs: Any, maximize: bool = False) -> Certificate | Verification:
    """
    Check `pairs` and, if optimal, return a `Certificate`; otherwise a failed
    `Verification` (with a negative-cycle witness in `reason` when suboptimal).
    """
    n, m, rows = _exact_rows(matrix, maximize)
    pairs = tuple((int(r), int(c)) for r, c in pairs)
    on, om, orows, tr = _orient(n, m, rows)
    mp = [(c, r) if tr else (r, c) for r, c in pairs]
    # primal feasibility (independent of any solver)
    if len(mp) != on or len({r for r, _ in mp}) != on or len({c for _, c in mp}) != on:
        return Verification(
            False, "primal", f"expected {on} pairs on distinct rows and columns, got {len(mp)}"
        )
    for r, c in mp:
        if not (0 <= r < on and 0 <= c < om) or c not in orows[r]:
            return Verification(False, "primal", f"pair {(r, c)} is out of range or forbidden")
    res = _padded_duals(on, om, orows, mp)
    if isinstance(res, Verification):
        return res
    uu, vv, _match, _size, _full = res
    free = [j for j in range(om) if j not in {c for _, c in mp}]
    w = Fraction(0)
    if free:
        w = vv[free[0]]  # all free columns share one dual value
        if any(vv[j] != w for j in free):
            return Verification(False, "internal", "free columns have unequal duals")
    u_real = tuple(uu[i] + w for i in range(on))
    v_real = tuple(vv[j] - w for j in range(om))
    total = sum((orows[r][c] for r, c in mp), Fraction(0))
    cert = Certificate(
        CERT_VERSION,
        problem_hash(matrix, maximize),
        (n, m),
        maximize,
        tuple(sorted(pairs)),
        -total if maximize else total,
        u_real,
        v_real,
        tr,
        {"verifier": "paldita-bellman-ford-v1"},
    )
    chk = cert.verify(matrix)
    return cert if chk else chk


def verify_pairs(matrix: Any, pairs: Any, maximize: bool = False) -> Verification:
    """True-ish iff `pairs` is a provably optimal assignment of `matrix`."""
    out = certify(matrix, pairs, maximize)
    if isinstance(out, Certificate):
        return Verification(True, "ok", "optimal", out.objective)
    return out


class CertificateCache:
    """Memoise `verify_pairs` by (problem hash, pairs, sense) so an identical proof obligation is checked once (LRU)."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize, self.hits, self.misses = maxsize, 0, 0
        self._d: dict[tuple[str, tuple[tuple[int, int], ...], bool], Verification] = {}

    def verify(self, matrix: Any, pairs: Any, maximize: bool = False) -> Verification:
        key = (
            problem_hash(matrix, maximize),
            tuple(sorted((int(r), int(c)) for r, c in pairs)),
            bool(maximize),
        )
        if key in self._d:
            self.hits += 1
            self._d[key] = self._d.pop(key)  # refresh recency
            return self._d[key]
        self.misses += 1
        out = verify_pairs(matrix, pairs, maximize)
        self._d[key] = out
        if len(self._d) > self.maxsize:
            self._d.pop(next(iter(self._d)))
        return out
