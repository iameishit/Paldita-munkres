#!/usr/bin/env python
"""
Differential + hostile-input fuzzer for the native kernel (`munkres._lapc`).

Run it against a sanitizer build (see tools/sanitize_native.sh) so that any out-of-bounds access,
misaligned load, signed overflow or use-after-free aborts the process:

    python tools/fuzz_native.py --iterations 20000 --seed 1

Valid inputs are checked against the exact pure-Python solver; hostile inputs (wrong item size,
misaligned or read-only buffers, truncated buffers, absurd sizes, NaN/inf/INT64_MAX data) must
raise a Python exception or return a status code, never crash.
"""

import argparse
import pathlib
import random
import sys
from array import array

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from munkres import _native
from munkres._api import solve

INF = float("inf")


def exact_total(rows, maximize=False):
    return solve(rows, maximize=maximize).total


def valid_cases(rng, mod, iterations):
    for _ in range(iterations):
        n = rng.randint(0, 6)
        m = rng.randint(n, 8)
        kind = rng.randint(0, 1)
        if kind == 0:
            rows = [
                [
                    rng.choice([float(rng.randint(-9, 9)), rng.random() * 8 - 4, INF])
                    for _ in range(m)
                ]
                for _ in range(n)
            ]
            flat = array("d", [x for r in rows for x in r])
        else:
            rows = [[rng.randint(-50, 50) for _ in range(m)] for _ in range(n)]
            flat = array("q", [x for r in rows for x in r])
        r2c = array("i", [-1]) * max(n, 1)
        hall = array("i", [0]) * (n + m + 4)
        rc = mod.solve(kind, n, m, flat, r2c, None, None, hall)
        if rc == 0:
            cols = list(r2c[:n])
            assert len(set(cols)) == n, (rows, cols)
            assert all(0 <= c < m for c in cols), (rows, cols)
            cells = [rows[i][c] for i, c in enumerate(cols)]
            assert all(x != INF for x in cells)
            if n and (
                kind == 1 or all(x != INF for r in rows for x in r)
            ):  # exact oracle: no forbidden cells in play
                want = exact_total(rows)
                assert abs(sum(cells) - float(want)) < 1e-9, (rows, cols, want)
        else:
            assert rc in (1, 2, 4), rc


def hostile_cases(rng, mod, iterations):
    for _ in range(iterations):
        n, m = rng.choice([(2, 2), (1, 4), (0, 0), (3, 3), (2, 5)])
        cells = n * m
        pick = rng.randint(0, 9)
        cost = array(
            "d", [rng.choice([1.0, float("nan"), INF, -INF, 1e308, -1e308]) for _ in range(cells)]
        )
        r2c = array("i", [0]) * max(n, 1)
        hall = array("i", [0]) * (n + m + 4)
        args = [0, n, m, cost, r2c, None, None, hall]
        if pick == 0:
            args[3] = array("f", [1.0]) * cells  # wrong item size
        elif pick == 1:
            args[3] = memoryview(bytearray(8 * cells + 1))[
                1:
            ]  # misaligned view of the right length
        elif pick == 2:
            args[3] = memoryview(bytes(8 * cells))  # read-only is fine for the input...
        elif pick == 3:
            args[4] = memoryview(bytes(4 * max(n, 1)))  # ...but not for the output
        elif pick == 4:
            args[3] = array("d", [1.0]) * max(cells - 1, 0)  # truncated
        elif pick == 5:
            args[7] = array("i", [0]) * max(n + m + 3, 0)  # hall one slot short
        elif pick == 6:
            args[1], args[2] = 2**31 - 1, 2**31 - 1  # absurd sizes
        elif pick == 7:
            args[0] = rng.choice([-1, 2, 99])  # bad kind
        elif pick == 8:
            args[5] = array("f", [0.0]) * (n + 1)  # u with wrong item size
        try:
            mod.solve(*args)
        except (TypeError, ValueError, OverflowError, BufferError, MemoryError):
            pass
    # batch entry point
    for _ in range(iterations // 4):
        B, n, m = rng.randint(0, 4), rng.randint(0, 3), rng.randint(0, 4)
        n, m = min(n, m), max(n, m)
        kind = rng.randint(0, 1)
        code = "d" if kind == 0 else "q"
        cost = array(code, [rng.choice([1, 2, 3]) for _ in range(B * n * m)])
        out = array("i", [0]) * max(B * n, 1)
        status = array("i", [0]) * max(B, 1)
        mod.solve_batch(B, n, m, cost, out, status, kind)
        try:
            mod.solve_batch(B + 1, n, m, cost, out, status, kind)  # buffers one problem too small
        except (ValueError, TypeError):
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mod = _native._load()
    if mod is None:
        print("native extension not built; nothing to fuzz")
        return 2
    rng = random.Random(args.seed)
    valid_cases(rng, mod, args.iterations)
    hostile_cases(rng, mod, args.iterations)
    print(
        f"native fuzz ok: {args.iterations} valid + {args.iterations} hostile cases, seed {args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
