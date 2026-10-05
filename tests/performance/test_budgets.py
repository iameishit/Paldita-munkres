"""Generous time and memory budgets: they only trip on a real algorithmic regression."""

import random
import time
import tracemalloc

from munkres import DISALLOWED, Munkres, solve


def random_matrix(n, seed, high=1000):
    rng = random.Random(seed)
    return [[rng.randint(0, high) for _ in range(n)] for _ in range(n)]


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def test_dense_300_finishes_quickly():
    assert timed(Munkres().compute, random_matrix(300, 1)) < 15


def test_many_ties_stay_fast():
    assert timed(Munkres().compute, random_matrix(300, 2, high=3)) < 10


def test_rectangular_problems_cost_by_the_smaller_side():
    rng = random.Random(3)
    wide = [[rng.randint(0, 1000) for _ in range(600)] for _ in range(40)]
    assert timed(Munkres().compute, wide) < 10
    assert timed(Munkres().compute, [list(col) for col in zip(*wide, strict=True)]) < 10


def test_sparse_problems_are_cheap():
    rng = random.Random(4)
    n = 300
    sparse = [
        [rng.randint(0, 99) if rng.random() < 0.05 or i == j else DISALLOWED for j in range(n)]
        for i in range(n)
    ]
    assert timed(solve, sparse) < 10


def test_growth_is_polynomial_not_explosive():
    small, large = (
        timed(Munkres().compute, random_matrix(80, 5)),
        timed(Munkres().compute, random_matrix(160, 5)),
    )
    assert large < 40 * max(small, 0.001)  # doubling n costs about 8x; 40x leaves wide margin


def test_peak_memory_is_proportional_to_the_matrix():
    matrix = random_matrix(250, 6)
    tracemalloc.start()
    Munkres().compute(matrix)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 80 * 1024 * 1024
