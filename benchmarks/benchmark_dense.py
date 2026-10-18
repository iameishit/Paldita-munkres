"""Square matrices of growing size  (`--quick` runs small sizes for a smoke test)."""

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
    rng = random.Random(0)
    print(f"{'n':>6} {'seconds':>10}")
    for n in (10, 20) if QUICK else (50, 100, 200, 300, 500):
        print(f"{n:>6} {timed(Munkres().compute, dense(rng, n, n)):>10.4f}")


if __name__ == "__main__":
    main()
