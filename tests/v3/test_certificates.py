"""Exact certificates and the independent verifier (with fault injection)."""

import random

from munkres import (
    DISALLOWED,
    Certificate,
    UnsolvableMatrix,
    certify,
    problem_hash,
    solve,
    verify_pairs,
)

from .helpers import all_optima


def _random(rng):
    r, c = rng.randint(1, 5), rng.randint(1, 5)
    return [
        [rng.randint(-5, 9) if rng.random() > 0.2 else DISALLOWED for _ in range(c)]
        for _ in range(r)
    ]


def test_solver_output_is_always_certified_and_json_round_trips():
    rng = random.Random(0)
    n = 0
    for _ in range(1200):
        a, mx = _random(rng), rng.random() < 0.3
        try:
            s = solve(a, maximize=mx)
        except UnsolvableMatrix:
            continue
        n += 1
        cert = certify(a, s.pairs, mx)
        assert isinstance(cert, Certificate), cert
        assert cert.objective == s.total
        assert Certificate.from_json(cert.to_json()).verify(a)
    assert n > 500


def test_verifier_rejects_every_corruption():
    rng = random.Random(1)
    rejected = 0
    for _ in range(1500):
        a, mx = _random(rng), rng.random() < 0.3
        try:
            s = solve(a, maximize=mx)
        except UnsolvableMatrix:
            continue
        cert = certify(a, s.pairs, mx)
        tampered_u = Certificate(
            cert.version,
            cert.problem_hash,
            cert.shape,
            cert.maximize,
            cert.pairs,
            cert.objective,
            tuple(x + 1 for x in cert.u),
            cert.v,
            cert.transposed,
            {},
        )
        assert not tampered_u.verify(a)
        tampered_obj = Certificate(
            cert.version,
            cert.problem_hash,
            cert.shape,
            cert.maximize,
            cert.pairs,
            cert.objective + 1,
            cert.u,
            cert.v,
            cert.transposed,
            {},
        )
        assert not tampered_obj.verify(a)
        if len(s.pairs) >= 2:
            (p, q), (r, t) = s.pairs[0], s.pairs[1]
            alt = [(p, t), (r, q), *s.pairs[2:]]
            if all(a[i][j] is not DISALLOWED for i, j in alt):
                equal = sum(a[i][j] for i, j in alt) == s.total
                assert bool(verify_pairs(a, alt, mx)) == equal  # accepted iff also optimal
                rejected += not equal
    assert rejected > 100


def test_suboptimal_pairs_report_negative_cycle():
    out = verify_pairs([[1, 5], [5, 1]], [(0, 1), (1, 0)])
    assert not out and out.stage == "optimality" and "cycle" in out.reason


def test_certificate_is_bound_to_its_problem():
    a, b = [[1, 2], [2, 1]], [[1, 2], [2, 9]]
    cert = certify(a, solve(a).pairs)
    assert cert.problem_hash == problem_hash(a) != problem_hash(b)
    assert not cert.verify(b)


def test_brute_force_agreement_on_objective():
    rng = random.Random(2)
    for _ in range(300):
        a = _random(rng)
        ref = all_optima(a)
        if ref is None:
            continue
        assert solve(a).total == ref[0]
