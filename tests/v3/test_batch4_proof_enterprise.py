import json
import random
from fractions import Fraction

import pytest

from munkres import (
    DISALLOWED,
    CostSourceRegistry,
    ProblemTooLargeError,
    SolverLimits,
    UnsolvableMatrix,
    choose_representation,
    crt_objective,
    estimate_density,
    estimate_memory,
    latency_profile,
    namespace_of,
    plan,
    rational_reconstruct,
    select_backend,
    solve,
    solve_limited,
    solve_spec,
    solve_with_proof,
    validate_passes,
    validate_transform,
    verify_proof,
)


def test_every_pass_validates_and_a_wrong_claim_is_caught():
    rng = random.Random(1)
    for _ in range(80):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        a = [
            [rng.randint(-5, 9) if rng.random() > 0.15 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        mx = rng.random() < 0.3
        for rec in validate_passes(a, mx):
            assert rec["ok"], rec
    a = [[4, 1], [2, 5]]
    # subtracting 1 from every cell of a 2x2 problem lowers any full assignment by 2: opt(a) = opt(b) + 2
    assert validate_transform(a, [[3, 0], [1, 4]], 2).ok
    assert (
        not validate_transform(a, [[3, 0], [1, 4]], 1).ok
        and not validate_transform(a, [[3, 0], [1, 4]], 5).ok
    )


def test_proof_graph_replays_and_rejects_tampering():
    import dataclasses

    a = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pf = solve_with_proof(a)
    assert verify_proof(a, pf).ok and [s.name for s in pf.steps] == [
        "input",
        "plan",
        "solve",
        "certify",
    ]
    assert not verify_proof([[4, 1, 3], [2, 0, 5], [3, 2, 9]], pf).ok  # different problem
    bad = dataclasses.replace(pf, pairs=((0, 0), (1, 1), (2, 2)))
    assert not verify_proof(a, bad).ok
    bad2 = dataclasses.replace(
        pf,
        steps=(pf.steps[0], dataclasses.replace(pf.steps[1], output_hash="0" * 64), *pf.steps[2:]),
    )
    assert not verify_proof(a, bad2).ok


def test_crt_objective_and_rational_reconstruction():
    rng = random.Random(2)
    for _ in range(200):
        n = rng.randint(1, 5)
        a = [[rng.randint(-(10**12), 10**12) for _ in range(n)] for _ in range(n)]
        pairs = solve(a).pairs
        assert crt_objective(a, pairs) == sum(a[i][j] for i, j in pairs)
    m = (2**61 - 1) * (2**31 - 1)
    for _ in range(200):
        p, q = rng.randint(-(10**6), 10**6), rng.randint(1, 10**6)
        f = Fraction(p, q)
        assert rational_reconstruct(f.numerator * pow(f.denominator, -1, m) % m, m) == f
    with pytest.raises(ValueError):
        rational_reconstruct(123456789123456789 % m, 1009 * 1013)


def test_select_backend_by_capability():
    assert select_backend(["duals"])["name"] == "python-exact"
    assert select_backend(["exact_arithmetic", "Fraction"])["name"] == "python-exact"
    assert select_backend(["float64"])["name"] in {"native-kernel", "python-exact"}
    with pytest.raises(LookupError):
        select_backend(["quantum_supremacy"])


def test_plan_obligations_force_the_exact_path_when_overflow_is_possible():
    p = plan([[1e308, 1.0], [1.0, 1e308]])
    assert p.strategy == "python-exact" and any(
        name.startswith("max|c|") and not ok for name, ok, _ in p.obligations
    )
    q = plan([[1, 2], [3, 4]])
    assert all(ok for _, ok, _ in q.obligations)
    from fractions import Fraction as F  # noqa: N817

    assert plan([[F(1), F(2)], [F(3), F(4)]]).strategy == "python-exact"  # not in the native domain


def test_limits_are_enforced_before_allocation_and_errors_are_structured():
    big = [[0] * 50 for _ in range(50)]
    with pytest.raises(ProblemTooLargeError) as e:
        solve_limited(big, SolverLimits(max_rows=10))
    assert (
        e.value.code == "ROWS_LIMIT"
        and e.value.details["rows"] == 50
        and isinstance(e.value, ValueError)
    )
    for lim, code in [
        (SolverLimits(max_cols=10), "COLS_LIMIT"),
        (SolverLimits(max_cells=100), "CELLS_LIMIT"),
        (SolverLimits(max_memory_bytes=1000), "MEMORY_LIMIT"),
    ]:
        with pytest.raises(ProblemTooLargeError) as e:
            solve_limited(big, lim)
        assert e.value.code == code
    assert solve_limited([[1, 2], [2, 1]], SolverLimits(max_rows=5)).total == 2

    class Shape:  # a "huge" array stand-in: only .shape is touched, nothing is allocated
        shape = (10**6, 10**6)

    with pytest.raises(ProblemTooLargeError):
        solve_limited(Shape(), SolverLimits(max_cells=10**9))
    assert estimate_memory(1000, 1000) == 8_000_000
    assert choose_representation(10**5, 10**5, 0.001)["representation"] == "sparse"
    assert (
        choose_representation(10**6, 10**6, 0.5, budget_bytes=1 << 20)["representation"] == "lazy"
    )
    assert choose_representation(100, 100, 1.0)["representation"] == "dense"


def test_declarative_spec_round_trips_and_validates():
    spec = {
        "cost": [[4, 1, None], [2, 0, 5], [3, 2, 2]],
        "objective": "min",
        "forbidden": [[2, 0]],
        "verify": True,
    }
    out = json.loads(json.dumps(solve_spec(spec)))
    assert out["total"] == solve([[4, 1, DISALLOWED], [2, 0, 5], [DISALLOWED, 2, 2]]).total
    assert (2, 0) not in {tuple(p) for p in out["pairs"]} and out["certificate"][
        "objective"
    ] == str(out["total"])
    assert solve_spec({"cost": [[1, 2], [3, 4]], "objective": "max"})["total"] == 5
    for bad in (
        {},
        {"cost": []},
        {"cost": [[1, 2], [3]]},
        {"cost": [[1]], "objective": "median"},
        {"cost": [["a"]]},
        {"cost": [[1]], "forbidden": [[5, 5]]},
        {"cost": [[1]], "nope": 1},
    ):
        with pytest.raises(ValueError):
            solve_spec(bad)
    with pytest.raises(ProblemTooLargeError):
        solve_spec({"cost": [[1, 2], [3, 4]], "limits": {"max_rows": 1}})
    with pytest.raises(UnsolvableMatrix):
        solve_spec({"cost": [[None, None], [1, 2]]})


def test_namespace_cost_sources_latency_and_density():
    ns = namespace_of([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert ns.ls("/") == ["backends", "certificate", "plan", "problem", "solution"]
    assert (
        ns.read("/solution/total") == 5
        and ns.read("/problem/shape") == (3, 3)
        and "native-kernel" in ns.ls("/backends")
    )
    with pytest.raises(KeyError):
        ns.read("/nope")
    with pytest.raises(IsADirectoryError):
        ns.read("/plan")
    reg = CostSourceRegistry()
    reg.mount("m", [[1, 2], [2, 1]])
    reg.mount("f", lambda i, j: abs(i - j), shape=(4, 4))
    assert reg.names() == ["f", "m"] and reg.solve("m").total == 2 and reg.solve("f").total == 0
    with pytest.raises(ValueError):
        reg.mount("bad", lambda i, j: 0)
    prof = latency_profile(lambda: solve([[1, 2], [2, 1]]), repeat=100, warmup=5)
    assert (
        prof["p50"] <= prof["p90"] <= prof["p95"] <= prof["p99"] <= prof["p999"] <= prof["max"]
        and prof["n"] == 100
    )
    m = [[1 if (i * 7 + j * 3) % 10 < 3 else DISALLOWED for j in range(50)] for i in range(50)]
    true = sum(v is not DISALLOWED for r in m for v in r) / 2500
    est, se = estimate_density(m, 4000, random.Random(1))
    assert abs(est - true) < 5 * se + 0.02
