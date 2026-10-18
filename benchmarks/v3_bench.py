#!/usr/bin/env python
"""
Reproducible v3 benchmark: Paldita vs SciPy on identical inputs (both timed end to end from a NumPy
array to the (row_ind, col_ind) result). Fixed seeds; best-of-repeats and p50/p95/p99 are reported;
environment is recorded. Writes benchmarks/results/v3_baseline.json.

    python benchmarks/v3_bench.py [--quick]
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment as sp_lsa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import munkres
from munkres import _native

QUICK = "--quick" in sys.argv


def timeit(fn, min_time=0.15, repeats=7):
    """Per-call seconds for each repeat (each repeat runs fn enough times to last >= min_time)."""
    fn()
    n = 1
    while True:
        t = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t
        if dt >= min_time / 4 or n > 1_000_000:
            break
        n *= 4
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        out.append((time.perf_counter() - t) / n)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def row(name, ours, theirs):
    a, b = min(ours), min(theirs)
    return {
        "case": name,
        "paldita_best_us": round(a * 1e6, 2),
        "scipy_best_us": round(b * 1e6, 2),
        "ratio_paldita_over_scipy": round(a / b, 3),
        "paldita_p50_us": round(statistics.median(ours) * 1e6, 2),
        "scipy_p50_us": round(statistics.median(theirs) * 1e6, 2),
        "paldita_p95_us": round(pct(ours, 95) * 1e6, 2),
        "scipy_p95_us": round(pct(theirs, 95) * 1e6, 2),
    }


def main():
    rng = np.random.default_rng(12345)
    res = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "native_kernel": _native.version(),
            "munkres": munkres.__version__,
            "seed": 12345,
            "quick": QUICK,
        },
        "dense": [],
        "batch": [],
        "dynamic": [],
        "notes": [
            "shared CI runners: absolute numbers are noisy; compare ratios, trust only large gaps",
            "both libraries timed from an in-memory float64 ndarray to (row_ind, col_ind)",
        ],
    }
    sizes = [5, 10, 32, 64, 128, 256, 512] if QUICK else [5, 10, 32, 64, 128, 256, 512, 1000, 2000]
    for n in sizes:
        a = rng.random((n, n))
        r = row(
            f"random float {n}x{n}",
            timeit(lambda: munkres.linear_sum_assignment(a)),
            timeit(lambda: sp_lsa(a)),
        )
        if (
            n <= 32
        ):  # proof-carrying record: the instance hash, agreement with SciPy and an independent exact certificate
            ours = munkres.linear_sum_assignment(a)
            theirs = sp_lsa(a)
            r["problem_hash"] = munkres.problem_hash(a.tolist())
            r["objective_agrees_with_scipy"] = bool(abs(a[ours].sum() - a[theirs].sum()) < 1e-9)
            r["certificate_verified"] = bool(
                munkres.verify_pairs(a.tolist(), list(zip(*[x.tolist() for x in ours])))
            )
        res["dense"].append(r)
    for n in (64, 256):
        a = rng.integers(0, 100, (n, n)).astype(np.int64)
        res["dense"].append(
            row(
                f"small-integer int64 {n}x{n} (ties)",
                timeit(lambda: munkres.linear_sum_assignment(a)),
                timeit(lambda: sp_lsa(a)),
            )
        )
        b = rng.random((n, 2 * n))
        res["dense"].append(
            row(
                f"rectangular float {n}x{2 * n}",
                timeit(lambda: munkres.linear_sum_assignment(b)),
                timeit(lambda: sp_lsa(b)),
            )
        )
        c = rng.random((n, n))
        c[rng.random((n, n)) < 0.9] = np.inf
        c[np.arange(n), rng.permutation(n)] = rng.random(n)  # keep it feasible
        res["dense"].append(
            row(
                f"90% forbidden float {n}x{n}",
                timeit(lambda: munkres.linear_sum_assignment(c)),
                timeit(lambda: sp_lsa(c)),
            )
        )
    for B, n in [(20000, 8), (5000, 16), (1000, 64)] if not QUICK else [(5000, 8)]:
        x = rng.random((B, n, n))
        ours = timeit(lambda: munkres.solve_batch(x), min_time=0.3, repeats=5)
        theirs = timeit(lambda: [sp_lsa(m) for m in x], min_time=0.3, repeats=5)
        r = row(f"batch {B} x {n}x{n} (per batch)", ours, theirs)
        r["paldita_solves_per_s"] = round(B / min(ours))
        r["scipy_loop_solves_per_s"] = round(B / min(theirs))
        res["batch"].append(r)
    # dynamic repair vs full exact re-solve (both exact rational arithmetic)
    import random

    rng2 = random.Random(7)
    n = 40
    a = [[rng2.randint(0, 100) for _ in range(n)] for _ in range(n)]
    st = munkres.AssignmentState(a)
    t0 = time.perf_counter()
    st.solve()
    full = time.perf_counter() - t0
    reaug, times, ok = [], [], True
    for _ in range(30 if not QUICK else 10):
        i, j, v = rng2.randrange(n), rng2.randrange(n), rng2.randint(0, 100)
        a[i][j] = v
        st.update_cost(i, j, v)
        t0 = time.perf_counter()
        st.solve()
        times.append(time.perf_counter() - t0)
        reaug.append(st.stats.reaugmented_rows)
        ok &= st.objective == munkres.solve(a).total
    res["dynamic"].append(
        {
            "case": "exact AssignmentState, 1 changed cell per step, 40x40",
            "full_exact_solve_ms": round(full * 1e3, 2),
            "repair_median_ms": round(statistics.median(times) * 1e3, 3),
            "repair_p95_ms": round(pct(times, 95) * 1e3, 3),
            "rows_reaugmented_mean": round(statistics.mean(reaug), 2),
            "rows_total": n,
            "always_matches_fresh_solve": bool(ok),
        }
    )
    path = os.path.join(os.path.dirname(__file__), "results", "v3_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for sec in ("dense", "batch"):
        for r in res[sec]:
            print(
                f"{r['case']:42s} paldita {r['paldita_best_us']:>11.1f} us  scipy {r['scipy_best_us']:>11.1f} us  ratio {r['ratio_paldita_over_scipy']:.2f}"
            )
    print(res["dynamic"][0])
    print("wrote", path)


if __name__ == "__main__":
    main()
