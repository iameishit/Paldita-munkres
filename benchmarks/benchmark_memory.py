"""Peak memory while solving  (`--quick` runs small sizes for a smoke test)."""

import random
import sys
import time
import tracemalloc

from munkres import Munkres

QUICK = "--quick" in sys.argv


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def dense(rng, rows, cols, high=1000):
    return [[rng.randint(0, high) for _ in range(cols)] for _ in range(rows)]


def peak_mb(matrix):
    tracemalloc.start()
    Munkres().compute(matrix)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def main():
    rng = random.Random(3)
    print(f"{'n':>6} {'peak MiB':>10}")
    for n in (10, 20) if QUICK else (50, 100, 200, 400):
        print(f"{n:>6} {peak_mb(dense(rng, n, n)):>10.2f}")


if __name__ == "__main__":
    main()
