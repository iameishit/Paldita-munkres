"""Related problems: stable matching, transportation, Sinkhorn."""

import itertools
import math
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from munkres import (
    DISALLOWED,
    UnsolvableMatrix,
    sinkhorn,
    soft_assignment,
    solve,
    stable_matching,
    transport,
)

from .helpers import all_matchings

D = DISALLOWED
COMMON = {
    "deadline": None,
    "database": None,
    "derandomize": True,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}


def matrices(max_rows=4, max_cols=4, *, forbid=False):
    cell = st.one_of(st.integers(-9, 9), st.just(D)) if forbid else st.integers(-9, 9)

    @st.composite
    def build(draw):
        r, c = draw(st.integers(1, max_rows)), draw(st.integers(1, max_cols))
        return [[draw(cell) for _ in range(c)] for _ in range(r)]

    return build()


def feasible(matrix):
    return next(all_matchings(matrix), None) is not None


# --------------------------------------------------------------------------
# stable matching
# --------------------------------------------------------------------------


def stable_matchings(proposers, receivers):
    """Every stable matching, by exhaustive enumeration (tiny instances)."""
    names = list(proposers)
    options = [[None, *proposers[p]] for p in names]
    for combo in itertools.product(*options):
        chosen = [r for r in combo if r is not None]
        if len(set(chosen)) != len(chosen):
            continue
        match = {p: r for p, r in zip(names, combo, strict=True) if r is not None}
        if any(p not in receivers[r] for p, r in match.items()):
            continue
        holder = {r: p for p, r in match.items()}

        def prefers(ranked, new, current):
            return current is None or ranked.index(new) < ranked.index(current)

        blocking = any(
            p in receivers[r]
            and r in proposers[p]
            and prefers(proposers[p], r, match.get(p))
            and prefers(receivers[r], p, holder.get(r))
            for p in names
            for r in receivers
        )
        if not blocking:
            yield match


@settings(max_examples=250, **COMMON)
@given(st.data())
def test_stable_matching_is_stable_and_proposer_optimal(data):
    n_p, n_r = data.draw(st.integers(1, 4)), data.draw(st.integers(1, 4))
    ps, rs = [f"p{i}" for i in range(n_p)], [f"r{i}" for i in range(n_r)]
    proposers = {
        p: data.draw(st.permutations(rs).map(lambda x: x[: data.draw(st.integers(0, n_r))]))
        for p in ps
    }
    receivers = {
        r: data.draw(st.permutations(ps).map(lambda x: x[: data.draw(st.integers(0, n_p))]))
        for r in rs
    }
    got = stable_matching(proposers, receivers)
    everything = list(stable_matchings(proposers, receivers))
    assert got in everything
    for p in ps:  # proposer-optimal: nobody could do better in any stable matching
        mine = got.get(p)
        for other in everything:
            theirs = other.get(p)
            if theirs is not None:
                assert mine is not None and proposers[p].index(mine) <= proposers[p].index(theirs)
    assert all(
        set(m) == set(got) for m in everything
    )  # same agents matched in all (rural hospitals)


def test_stable_matching_classic_example_and_validation():
    men = {"A": ["x", "y", "z"], "B": ["y", "x", "z"], "C": ["x", "y", "z"]}
    women = {"x": ["B", "A", "C"], "y": ["C", "A", "B"], "z": ["A", "B", "C"]}
    # hand-checked: no blocking pair (A would rather have x or y, but x prefers B and y prefers C)
    assert stable_matching(men, women) == {"A": "z", "B": "x", "C": "y"}
    flipped = stable_matching(women, men)  # women-proposing = receiver-optimal for the men
    assert flipped in list(stable_matchings(women, men))
    with pytest.raises(ValueError, match="twice"):
        stable_matching({"a": ["x", "x"]}, {"x": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        stable_matching({"a": ["nobody"]}, {"x": ["a"]})
    with pytest.raises(ValueError, match="receiver"):
        stable_matching({"a": ["x"]}, {"x": ["ghost"]})


# --------------------------------------------------------------------------
# transportation problem (oracle: scipy's LP solver)
# --------------------------------------------------------------------------


def lp_transport_cost(supply, demand, cost):
    np = pytest.importorskip("numpy")
    opt = pytest.importorskip("scipy.optimize")
    n, m = len(supply), len(demand)
    c = np.array(cost, dtype=float).ravel()
    a_ub, b_ub = [], []
    for i in range(n):  # row sums <= supply
        row = np.zeros(n * m)
        row[i * m : (i + 1) * m] = 1
        a_ub.append(row)
        b_ub.append(supply[i])
    for j in range(m):  # column sums <= demand
        col = np.zeros(n * m)
        col[j::m] = 1
        a_ub.append(col)
        b_ub.append(demand[j])
    shipped = min(sum(supply), sum(demand))
    res = opt.linprog(c, A_ub=np.array(a_ub), b_ub=b_ub, A_eq=[np.ones(n * m)], b_eq=[shipped],
                      bounds=(0, None), method="highs")  # fmt: skip
    return res.fun


def test_transport_matches_an_lp_solver_on_random_instances():
    rng = random.Random(3)
    for _ in range(60):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        supply = [rng.randint(0, 5) for _ in range(n)]
        demand = [rng.randint(0, 5) for _ in range(m)]
        if sum(supply) == 0 or sum(demand) == 0:
            continue
        cost = [[rng.randint(1, 20) for _ in range(m)] for _ in range(n)]
        plan = transport(supply, demand, cost)
        assert plan.total_cost == pytest.approx(lp_transport_cost(supply, demand, cost))
        assert plan.shipped == min(sum(supply), sum(demand))
        for i in range(n):
            assert sum(u for (a, _), u in plan.flows.items() if a == i) <= supply[i]
        for j in range(m):
            assert sum(u for (_, b), u in plan.flows.items() if b == j) <= demand[j]


def test_transport_examples_and_errors():
    plan = transport([30 // 10, 2], [2, 3], [[1, 5], [4, 2]])
    assert plan.flows == {(0, 0): 2, (0, 1): 1, (1, 1): 2} and plan.total_cost == 2 + 5 + 4
    assert transport([0, 0], [1], [[1], [1]]).shipped == 0
    assert transport([2], [2], [[D]] if False else [[1]]).total_cost == 2
    with pytest.raises(UnsolvableMatrix, match="forbidden routes"):
        transport([1, 1], [2], [[D], [D]])
    for bad in ([-1, 2], [1.5, 2], [True, 1]):
        with pytest.raises(ValueError, match="non-negative integers"):
            transport(bad, [1], [[1], [1]])
    with pytest.raises(ValueError, match="cost must be"):
        transport([1], [1, 1], [[1]])
    with pytest.raises(ValueError, match="max_units"):
        transport([2000], [2000], [[1]])


# --------------------------------------------------------------------------
# Sinkhorn / soft assignment (oracle: an independent numpy/scipy implementation)
# --------------------------------------------------------------------------


def reference_sinkhorn(cost, reg, a, b, iterations=2000):
    np = pytest.importorskip("numpy")
    special = pytest.importorskip("scipy.special")
    c = np.array(cost, dtype=float)
    la, lb = np.log(a), np.log(b)
    f, g = np.zeros(len(a)), np.zeros(len(b))
    for _ in range(iterations):
        f = reg * (la - special.logsumexp((g[None, :] - c) / reg, axis=1))
        g = reg * (lb - special.logsumexp((f[:, None] - c) / reg, axis=0))
    return np.exp((f[:, None] + g[None, :] - c) / reg)


def test_sinkhorn_matches_an_independent_implementation():
    np = pytest.importorskip("numpy")
    rng = random.Random(8)
    for n, m, reg in ((4, 4, 0.5), (3, 5, 0.2), (5, 3, 1.0)):
        cost = [[rng.random() * 3 for _ in range(m)] for _ in range(n)]
        a = [rng.random() + 0.5 for _ in range(n)]
        b = [x * sum(a) / (m * 1.0) for x in [1.0] * m]  # equal masses summing to sum(a)
        result = sinkhorn(cost, reg=reg, a=a, b=b)
        assert result.converged
        assert np.allclose(np.array(result.plan), reference_sinkhorn(cost, reg, a, b), atol=1e-7)
        assert np.allclose(np.array(result.plan).sum(axis=1), a, atol=1e-8)
        assert np.allclose(np.array(result.plan).sum(axis=0), b, atol=1e-8)


def test_soft_assignment_sharpens_to_the_hungarian_answer():
    m = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    soft = soft_assignment(m, temperature=0.05)
    for i in range(3):
        assert soft[i][i] > 0.99 and sum(soft[i]) == pytest.approx(1)
    blurry = soft_assignment(m, temperature=5.0)  # high temperature = nearly uniform
    assert all(abs(x - 1 / 3) < 0.1 for row in blurry for x in row)
    exact = solve(m).total
    assert exact <= sinkhorn(m, reg=0.05).cost * 3 <= exact + 0.2  # cost converges to the optimum


def test_sinkhorn_forbidden_cells_get_no_mass_and_rectangles_work():
    result = sinkhorn([[0.0, D, 1.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]], reg=0.3)
    assert result.plan[0][1] == 0.0 and result.converged
    # a pattern that forces some cell to *exactly* zero mass is approached only
    # asymptotically, so it reports converged=False instead of looping forever
    assert sinkhorn([[0.0, D], [1.0, 2.0]], reg=0.3, max_iter=50).converged is False
    tall = soft_assignment([[0.0], [1.0], [2.0]], temperature=0.5)
    assert sum(tall[0]) == pytest.approx(1) and len(tall) == 3


def test_sinkhorn_validation():
    ok = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="empty"):
        sinkhorn([])
    for reg in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="reg"):
            sinkhorn(ok, reg=reg)
    with pytest.raises(ValueError, match="entries"):
        sinkhorn(ok, a=[1.0])
    with pytest.raises(ValueError, match="positive"):
        sinkhorn(ok, a=[0.5, 0.0])
    with pytest.raises(ValueError, match="total mass"):
        sinkhorn(ok, a=[0.5, 0.5], b=[1.0, 1.0])
    with pytest.raises(ValueError, match="allowed cell"):
        sinkhorn([[D, D], [1.0, 2.0]])
    lopsided = sinkhorn(
        [[0.0, 3.0, 1.0], [2.0, 0.0, 4.0], [5.0, 1.0, 0.0]],
        reg=0.5, a=[0.2, 0.3, 0.5], b=[0.4, 0.4, 0.2], max_iter=1,
    )  # fmt: skip
    assert lopsided.converged is False and lopsided.iterations == 1
    with pytest.raises(ValueError, match="did not converge"):
        soft_assignment(
            [[0.0, D], [1.0, 2.0]], temperature=0.3, max_iter=50
        )  # forces an exact zero
