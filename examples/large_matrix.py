"""A larger problem, with timing."""

import random
import time

from munkres import solve

rng = random.Random(0)
n = 300
costs = [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]
start = time.perf_counter()
result = solve(costs)
print(f"{n}x{n}: total {result.total} in {time.perf_counter() - start:.2f}s")
