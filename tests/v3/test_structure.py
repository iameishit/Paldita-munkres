import random

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    components,
    fingerprint,
    inverse_monge_witness,
    is_additive,
    monge_witness,
    plan,
    solve,
    solve_by_components,
    solve_monge,
    solve_planned,
)

from .helpers import all_optima


def _monge_like(rng):
    n = rng.randint(1, 4)
    m = rng.randint(n, 5)
    a = [rng.randint(-3, 3) for _ in range(n)]
    b = [rng.randint(-3, 3) for _ in range(m)]
    x = sorted(rng.randint(0, 4) for _ in range(n))
    y = sorted(rng.randint(0, 4) for _ in range(m))
    kind = rng.random()
    sign = -1 if kind < 0.4 else (1 if kind < 0.7 else 0)
    if sign == 0 and kind >= 0.7:
        return [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    return [[a[i] + b[j] + sign * x[i] * y[j] for j in range(m)] for i in range(n)]


def test_monge_solver_is_optimal_and_witnesses_are_sound():
    rng = random.Random(5)
    used = 0
    for _ in range(1500):
        a = _monge_like(rng)
        pairs, method = solve_monge(a)
        used += method != "general"
        assert sum(a[i][j] for i, j in pairs) == all_optima(a)[0] and len(pairs) == len(a)
        w = monge_witness(a)
        if w is not None:
            i, j = w
            assert a[i][j] + a[i + 1][j + 1] > a[i][j + 1] + a[i + 1][j]
    assert used > 800


def test_additive_and_forbidden_cells():
    assert is_additive([[1, 2], [3, 4]])
    # contract: a matrix with a forbidden cell cannot be proved Monge, and is reported at (0, 0)
    assert monge_witness([[1, DISALLOWED], [2, 3]]) == (0, 0)
    assert inverse_monge_witness([[1, DISALLOWED], [2, 3]]) == (0, 0)
    assert not is_additive([[1, DISALLOWED], [2, 3]])
    assert (
        monge_witness([[1, 2], [3, 4]]) is None
    )  # the finite twin of that matrix *is* provably Monge
    assert inverse_monge_witness([[2, 1], [1, 2]]) is None  # 2+2 >= 1+1
    assert inverse_monge_witness([[1, 2], [2, 1]]) == (0, 0)  # 1+1 < 2+2


def test_components_match_global_solution():
    rng = random.Random(6)
    for _ in range(1000):
        a = _monge_like(rng)
        b = [[v if rng.random() > 0.45 else DISALLOWED for v in row] for row in a]
        try:
            ref = solve(b).total
        except UnsolvableMatrix:
            ref = None
        try:
            _, total, _ = solve_by_components(b)
        except UnsolvableMatrix:
            total = None
        assert ref == total


def test_fingerprint_and_plan_explain():
    fp = fingerprint([[1, DISALLOWED], [DISALLOWED, 2]])
    assert fp.components == 2 and fp.density == 0.5
    assert fp.shape == (2, 2) and fp.allowed_cells == 2
    assert (fp.min_cost, fp.max_cost) == (1, 2)
    assert fp.isolated_rows == 0 and fp.isolated_columns == 0
    assert (
        not fp.monge and not fp.inverse_monge and not fp.additive
    )  # forbidden cells: nothing is proved
    assert any("forbidden" in note for note in fp.notes)
    p = plan([[1, DISALLOWED], [DISALLOWED, 2]])
    assert p.strategy == "components" and "independent components" in p.explain()
    assert components([[1, DISALLOWED], [DISALLOWED, 2]]) == [([0], [0]), ([1], [1])]


def test_planned_result_is_verified_optimal():
    rng = random.Random(8)
    for _ in range(200):
        a = [[rng.randint(0, 9) for _ in range(4)] for _ in range(4)]
        r = solve_planned(a, verify=True)
        assert r.certificate.verify(a) and r.total == all_optima(a)[0]
