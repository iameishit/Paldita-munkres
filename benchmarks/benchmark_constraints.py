"""Forbidden cells and gating  (`--quick` runs small sizes for a smoke test)."""

import random
import sys
import time

from munkres import DISALLOWED, solve

QUICK = "--quick" in sys.argv


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def dense(rng, rows, cols, high=1000):
    return [[rng.randint(0, high) for _ in range(cols)] for _ in range(rows)]


def main():
    rng = random.Random(2)
    n = 30 if QUICK else 200
    print(f"{'case':>26} {'seconds':>10}")
    for density in (0.0, 0.5, 0.9):
        matrix = [
            [
                DISALLOWED if i != j and rng.random() < density else rng.randint(0, 1000)
                for j in range(n)
            ]
            for i in range(n)
        ]
        print(f"{f'{int(density * 100)}% forbidden':>26} {timed(solve, matrix):>10.4f}")
    matrix = dense(rng, n, n)
    for limit in (100, 500):
        print(f"{f'gating max_cost={limit}':>26} {timed(solve, matrix, max_cost=limit):>10.4f}")


if __name__ == "__main__":
    main()
