import random
from fractions import Fraction

from munkres import DISALLOWED, UnsolvableMatrix, edge_margins, solve, topology

from .helpers import all_optima


def test_topology_matches_enumeration_of_all_optima():
    rng = random.Random(11)
    degenerate = 0
    for _ in range(1200):
        n, m = rng.randint(1, 4), rng.randint(1, 5)
        a = [
            [rng.randint(0, 3) if rng.random() > 0.15 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        mx = rng.random() < 0.3
        ref = all_optima(a, mx)
        if ref is None:
            continue
        best, opts = ref
        t = topology(a, maximize=mx)
        inter, union = frozenset.intersection(*opts), frozenset.union(*opts)
        degenerate += len(opts) > 1
        assert t.forced == inter and (t.forced | t.optional) == union
        assert t.n_optimal == len(opts) and t.objective == Fraction(best)
        assert t.unique == (len(opts) == 1)
    assert degenerate > 100


def test_edge_margins_are_exact():
    a = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    base = solve(a).total
    for (i, j), d in edge_margins(a).items():
        alt = [r[:] for r in a]
        alt[i][j] = DISALLOWED
        try:
            ex = solve(alt).total - base
        except UnsolvableMatrix:
            ex = None
        assert d["exclude_penalty"] == ex
        sub = [[a[x][y] for y in range(3) if y != j] for x in range(3) if x != i]
        assert d["force_penalty"] == a[i][j] + solve(sub).total - base
