"""munkres 2 against SciPy  (`--quick` runs small sizes for a smoke test)."""

import random
import sys
import time

from munkres import Munkres, linear_sum_assignment

QUICK = "--quick" in sys.argv


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def dense(rng, rows, cols, high=1000):
    return [[rng.randint(0, high) for _ in range(cols)] for _ in range(rows)]


def main():
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment as scipy_lsa
    except ImportError:
        print("SciPy is not installed; nothing to compare against.")
        return
    rng = random.Random(5)
    print(f"{'n':>6} {'munkres':>10} {'munkres lsa':>12} {'scipy':>10}")
    for n in (10, 20) if QUICK else (50, 100, 200, 500):
        matrix = dense(rng, n, n)
        array = np.array(matrix)
        print(
            f"{n:>6} {timed(Munkres().compute, matrix):>10.4f}"
            f" {timed(linear_sum_assignment, matrix):>12.4f} {timed(scipy_lsa, array):>10.4f}"
        )


if __name__ == "__main__":
    main()
