"""Empirical growth: fits time ~ n^k  (`--quick` runs small sizes for a smoke test)."""

import math
import random
import sys
import time

from munkres import Munkres

QUICK = "--quick" in sys.argv


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def dense(rng, rows, cols, high=1000):
    return [[rng.randint(0, high) for _ in range(cols)] for _ in range(rows)]


def main():
    rng = random.Random(4)
    sizes = (10, 20, 40) if QUICK else (50, 100, 200, 400)
    times = [timed(Munkres().compute, dense(rng, n, n)) for n in sizes]
    print(f"{'n':>6} {'seconds':>10}")
    for n, t in zip(sizes, times, strict=True):
        print(f"{n:>6} {t:>10.4f}")
    slope = (math.log(times[-1]) - math.log(times[0])) / (math.log(sizes[-1]) - math.log(sizes[0]))
    print(f"fitted exponent k = {slope:.2f}  (time ~ n^k)")


if __name__ == "__main__":
    main()
