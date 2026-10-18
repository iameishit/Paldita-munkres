import itertools
import random

import pytest

from munkres import (
    DISALLOWED,
    SEMIRINGS,
    count_matchings,
    dm_blocks,
    is_circulant,
    is_toeplitz,
    is_totally_monotone,
    is_totally_unimodular,
    is_tree_metric,
    is_ultrametric,
    kronecker_factors,
    leftmost_row_minima,
    monge_violations,
    permanent,
    row_minima_monotone,
    semiring_permanent,
    tropical_permanent,
    tropical_rank,
)


def _leibniz(a):
    n = len(a)
    tot = 0
    for p in itertools.permutations(range(n)):
        sgn = 1
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    sgn = -sgn
        term = sgn
        for i in range(n):
            term *= a[i][p[i]]
        tot += term
    return tot


def _tu_brute(a):
    n, m = len(a), len(a[0])
    for k in range(1, min(n, m) + 1):
        for rs in itertools.combinations(range(n), k):
            for cs in itertools.combinations(range(m), k):
                if _leibniz([[a[i][j] for j in cs] for i in rs]) not in (-1, 0, 1):
                    return False
    return True


def test_total_unimodularity_matches_leibniz_brute_force():
    rng = random.Random(1)
    yes = no = 0
    for _ in range(300):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        a = [[rng.choice([-1, 0, 0, 1]) for _ in range(m)] for _ in range(n)]
        got = is_totally_unimodular(a)
        assert got == _tu_brute(a)
        yes += got
        no += not got
    assert yes > 30 and no > 30
    n = 3  # the assignment polytope's constraint matrix (rows/cols incidence of K_{3,3}) is TU
    inc = [
        [
            1 if (c < n and c == i) or (c >= n and c - n == j) else 0
            for i in range(n)
            for j in range(n)
        ]
        for c in range(2 * n)
    ]
    assert is_totally_unimodular(inc, max_dim=9)
    assert not is_totally_unimodular([[1, 1], [-1, 1]])


def test_kronecker_toeplitz_circulant_ultrametric_tree_metric():
    rng = random.Random(2)
    for _ in range(60):
        a = [[rng.randint(1, 4) for _ in range(2)] for _ in range(2)]
        b = [[rng.randint(1, 4) for _ in range(3)] for _ in range(2)]
        c = [[a[i // 2][j // 3] * b[i % 2][j % 3] for j in range(6)] for i in range(4)]
        found = kronecker_factors(c)
        assert found and any(len(fa) == 2 and len(fa[0]) == 2 for fa, _ in found)
        for fa, fb in found:  # every reported factorisation really multiplies back
            p, q, r, s = len(fa), len(fa[0]), len(fb), len(fb[0])
            assert all(
                fa[i // r][j // s] * fb[i % r][j % s] == c[i][j]
                for i in range(p * r)
                for j in range(q * s)
            )
    assert kronecker_factors([[1, 2, 3], [4, 5, 7]]) == []
    assert is_toeplitz([[1, 2, 3], [4, 1, 2], [5, 4, 1]]) and not is_toeplitz([[1, 2], [3, 5]])
    assert is_toeplitz([[1, 2, 3], [4, 1, 2]]) and not is_toeplitz([[1, 2], [3, 1], [4, 5]])
    assert is_circulant([[1, 2, 3], [3, 1, 2], [2, 3, 1]]) and not is_circulant([[1, 2], [1, 2]])
    assert is_ultrametric([[0, 1, 2], [1, 0, 2], [2, 2, 0]]) and not is_ultrametric(
        [[0, 1, 3], [1, 0, 1], [3, 1, 0]]
    )
    # path lengths in a weighted tree satisfy the four-point condition; a perturbed matrix does not
    for _ in range(40):
        n = 6
        parent = [None] + [rng.randrange(i) for i in range(1, n)]
        w = [0] + [rng.randint(1, 5) for _ in range(1, n)]
        depth = [0] * n
        anc = [[] for _ in range(n)]
        for i in range(1, n):
            depth[i] = depth[parent[i]] + w[i]
            anc[i] = anc[parent[i]] + [parent[i]]

        def dist(x, y):
            ax, ay = anc[x] + [x], anc[y] + [y]
            common = max(depth[c] for c in set(ax) & set(ay))
            return depth[x] + depth[y] - 2 * common

        d = [[dist(x, y) for y in range(n)] for x in range(n)]
        assert is_tree_metric(d)
    assert is_tree_metric([[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]])  # a star tree
    c4 = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]  # the 4-cycle metric is not a tree metric
    assert not is_tree_metric(c4)


def test_monge_violations_and_total_monotonicity_and_dc_row_minima():
    assert (
        monge_violations([[2, 1], [1, 2]])["fraction"] == 1
    )  # 2+2 > 1+1: violates the Monge inequality
    assert monge_violations([[1, 2], [3, 4]])["violations"] == []  # 1+4 <= 2+3
    assert monge_violations([[1, 2], [3, 5]])["violations"] == [(0, 0, 1)]  # 1+5 > 2+3 by 1
    rng = random.Random(3)
    tm = 0
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        x, y = (
            sorted(rng.randint(0, 5) for _ in range(n)),
            sorted(rng.randint(0, 5) for _ in range(m)),
        )
        a = [
            [-x[i] * y[j] + rng.randint(0, 0) for j in range(m)] for i in range(n)
        ]  # Monge => totally monotone
        assert is_totally_monotone(a)
        assert row_minima_monotone(a) == leftmost_row_minima(a)
        tm += 1
        r = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        if is_totally_monotone(r):
            assert row_minima_monotone(r) == leftmost_row_minima(r)
        else:
            with pytest.raises(ValueError):
                row_minima_monotone(r)
    assert tm == 300


def test_tropical_rank_and_semirings():
    assert tropical_rank([[0, 5], [5, 0]]) == 2
    assert tropical_rank([[1, 2], [2, 3]]) == 1  # a_i + b_j: every assignment ties
    rng = random.Random(4)
    for _ in range(300):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        a = [
            [rng.randint(0, 6) if rng.random() > 0.2 else DISALLOWED for _ in range(m)]
            for _ in range(n)
        ]
        ones = [[1 if v is not DISALLOWED else DISALLOWED for v in r] for r in a]
        assert semiring_permanent(a, "min_plus") == (
            tropical_permanent(a) if tropical_permanent(a) is not None else float("inf")
        )
        assert semiring_permanent(ones, "counting") == count_matchings(a) == permanent(ones)
        assert semiring_permanent(ones, "boolean") == (count_matchings(a) > 0)
        assert semiring_permanent(a, "max_plus") == (
            tropical_permanent(a, True) if count_matchings(a) else -float("inf")
        )
    assert set(SEMIRINGS) == {"min_plus", "max_plus", "counting", "boolean", "max_times"}


def test_dm_blocks_cells_are_exactly_those_in_some_perfect_matching():
    rng = random.Random(5)
    seen = 0
    for _ in range(500):
        n = rng.randint(1, 5)
        a = [[1 if rng.random() > 0.45 else DISALLOWED for _ in range(n)] for _ in range(n)]
        perfect = [
            p
            for p in itertools.permutations(range(n))
            if all(a[i][p[i]] is not DISALLOWED for i in range(n))
        ]
        if not perfect:
            with pytest.raises(ValueError):
                dm_blocks(a)
            continue
        seen += 1
        union = {(i, p[i]) for p in perfect for i in range(n)}
        got = dm_blocks(a)
        assert got.allowed_cells == union
        assert sorted(r for rs, _ in got.blocks for r in rs) == list(
            range(n)
        )  # blocks partition rows and columns
    assert seen > 100
