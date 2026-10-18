"""Wide and tall matrices  (`--quick` runs small sizes for a smoke test)."""

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
    rng = random.Random(1)
    shapes = ((10, 40), (40, 10)) if QUICK else ((50, 500), (500, 50), (100, 1000), (1000, 100))
    print(f"{'shape':>12} {'seconds':>10}")
    for rows, cols in shapes:
        print(f"{rows:>5}x{cols:<6} {timed(Munkres().compute, dense(rng, rows, cols)):>10.4f}")


if __name__ == "__main__":
    main()
