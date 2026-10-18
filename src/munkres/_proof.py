# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Proof graphs, pass validation, modular certificates and capability-based backend selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from munkres._core import DISALLOWED, _as_lists

__all__ = [
    "ProofGraph",
    "ProofStep",
    "crt_objective",
    "rational_reconstruct",
    "select_backend",
    "solve_with_proof",
    "validate_passes",
    "validate_transform",
    "verify_proof",
]


def validate_transform(
    original: Any, transformed: Any, constant: Any = 0, sign: int = 1, maximize: bool = False
) -> Any:
    """Exactly check  opt(original) == sign * opt(transformed) + constant  by solving both (the claim a preprocessing pass makes)."""
    from munkres._api import solve
    from munkres._verify import Verification

    lhs = Fraction(solve(original, maximize=maximize).total)
    rhs = sign * Fraction(
        solve(transformed, maximize=maximize if sign == 1 else not maximize).total
    ) + Fraction(constant)
    return Verification(
        lhs == rhs, "transform", f"opt(original)={lhs}, sign*opt(transformed)+constant={rhs}"
    )


def validate_passes(matrix: Any, maximize: bool = False) -> list[dict[str, Any]]:
    """Independently validate every preprocessing pass of this package on `matrix` (negate, transpose, reduce_costs, components)."""
    from munkres._api import solve
    from munkres._components import solve_by_components
    from munkres._core import UnsolvableMatrix
    from munkres._landscape import reduce_costs

    rows = _as_lists(matrix)
    out = []

    def rec(name: str, ok: bool, detail: str) -> None:
        out.append({"pass": name, "ok": bool(ok), "detail": detail})

    try:
        base = Fraction(solve(rows, maximize=maximize).total)
    except UnsolvableMatrix:
        return [{"pass": "solve", "ok": True, "detail": "infeasible instance; no pass to validate"}]  # nosec B105
    neg = [[x if x is DISALLOWED else -x for x in r] for r in rows]
    v = validate_transform(rows, neg, 0, -1, maximize)
    rec("negate", v.ok, v.reason)
    tr = [list(c) for c in zip(*rows, strict=False)]
    t = Fraction(solve(tr, maximize=maximize).total)
    rec("transpose", t == base, f"{t} vs {base}")
    if not maximize:
        red = reduce_costs(rows)
        v = validate_transform(rows, red.matrix, red.constant)
        rec("reduce_costs", v.ok, v.reason)
    c = Fraction(solve_by_components(rows, maximize)[1])
    rec("components", c == base, f"{c} vs {base}")
    return out


@dataclass(frozen=True)
class ProofStep:
    name: str
    input_hash: str
    output_hash: str
    claim: str


@dataclass(frozen=True)
class ProofGraph:
    """Ordered steps from the original problem to a certified answer; `verify_proof` replays every claim."""

    steps: tuple[ProofStep, ...]
    pairs: tuple[tuple[int, int], ...]
    certificate_json: str
    maximize: bool


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def solve_with_proof(matrix: Any, maximize: bool = False) -> ProofGraph:
    """Solve and return the chain: input -> (plan) -> solve -> certify, each step hashed so it can be replayed."""
    from munkres._plan import solve_planned
    from munkres._verify import problem_hash

    rows = _as_lists(matrix)
    res = solve_planned(rows, maximize, verify=True)
    ph = problem_hash(rows, maximize)
    steps = [
        ProofStep(
            "input",
            ph,
            ph,
            f"problem {len(rows)}x{len(rows[0]) if rows else 0}, maximize={maximize}",
        ),
        ProofStep(
            "plan",
            ph,
            _h(res.plan.strategy),
            f"strategy {res.plan.strategy}: " + "; ".join(res.plan.reasons),
        ),
        ProofStep("solve", _h(res.plan.strategy), _h(list(res.pairs)), f"pairs {list(res.pairs)}"),
        ProofStep(
            "certify",
            _h(list(res.pairs)),
            _h(res.certificate.to_json()),
            "exact duals re-derived independently",
        ),
    ]
    return ProofGraph(tuple(steps), res.pairs, res.certificate.to_json(), maximize)


def verify_proof(matrix: Any, proof: ProofGraph) -> Any:
    """Re-derive the problem hash, step-hash chain and certificate from scratch; any tampering fails."""
    from munkres._verify import Certificate, Verification, problem_hash

    rows = _as_lists(matrix)
    if problem_hash(rows, proof.maximize) != proof.steps[0].input_hash:
        return Verification(False, "proof-input", "proof was issued for a different problem")
    for a, b in zip(proof.steps, proof.steps[1:], strict=False):
        if a.output_hash != b.input_hash:
            return Verification(False, "proof-chain", f"step {b.name} does not follow {a.name}")
    if proof.steps[2].output_hash != _h(list(proof.pairs)) or proof.steps[3].output_hash != _h(
        proof.certificate_json
    ):
        return Verification(
            False, "proof-hash", "pairs or certificate differ from what the steps recorded"
        )
    cert = Certificate.from_json(proof.certificate_json)
    if sorted(cert.pairs) != sorted(proof.pairs):
        return Verification(False, "proof-pairs", "certificate pairs differ from the solution")
    return cert.verify(rows)


_PRIMES = (2_147_483_647, 2_147_483_629, 2_147_483_587, 2_147_483_579)


def crt_objective(matrix: Any, pairs: Any, primes: Any = _PRIMES) -> int:
    """Integer objective of `pairs` reconstructed only from residues modulo several primes (Chinese remainder theorem)."""
    rows = _as_lists(matrix)
    if any(not isinstance(rows[i][j], int) for i, j in pairs):
        raise ValueError("integer costs required")
    mod = 1
    x = 0
    for p in primes:
        r = sum(rows[i][j] % p for i, j in pairs) % p
        t = ((r - x) * pow(mod, -1, p)) % p
        x += mod * t
        mod *= p
    return x - mod if x > mod // 2 else x


def rational_reconstruct(a: int, m: int) -> Fraction:
    """The unique p/q with |p|, q <= sqrt(m/2) and p == a*q (mod m) (Wang's algorithm); ValueError if none."""
    bound = int((m // 2) ** 0.5)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        raise ValueError("no rational reconstruction")
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)


def select_backend(needs: list[str]) -> dict[str, Any]:
    """First available backend whose capability descriptor satisfies every need (a boolean flag name or a dtype name); LookupError otherwise."""
    from munkres._plan import backends

    for b in backends():
        if not b["available"]:
            continue
        if all((b.get(n) is True) or n in b["dtypes"] for n in needs):
            return b
    raise LookupError(f"no available backend supports {needs}")
