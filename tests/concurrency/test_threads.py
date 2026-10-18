"""Shared use from many threads and processes."""

import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from munkres import DISALLOWED, Munkres, k_best, solve

from ..reference.munkres_reference import min_cost_assignment


def make_matrices(count, seed):
    rng = random.Random(seed)
    return [[[rng.randint(0, 50) for _ in range(7)] for _ in range(7)] for _ in range(count)]


def test_one_shared_instance_across_many_threads():
    matrices = make_matrices(48, seed=1)
    shared = Munkres()

    def run(m):
        pairs = shared.compute(m)
        return sum(m[r][c] for r, c in pairs)

    with ThreadPoolExecutor(max_workers=8) as pool:
        totals = list(pool.map(run, matrices * 4))
    assert totals == [min_cost_assignment(m)[0] for m in matrices] * 4


def test_high_level_functions_are_thread_safe_too():
    matrices = make_matrices(16, seed=2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        solved = list(pool.map(solve, matrices))
        ranked = list(pool.map(lambda m: k_best(m, 3), matrices))
    for m, one, three in zip(matrices, solved, ranked, strict=True):
        assert one.total == three[0].total == min_cost_assignment(m)[0]


def test_matrices_with_disallowed_cross_process_boundaries():
    matrices = [[[1, DISALLOWED, 3], [2, 4, DISALLOWED], [DISALLOWED, 5, 6]]] * 4
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(solve, matrices))
    assert [r.total for r in results] == [min_cost_assignment(matrices[0])[0]] * 4
