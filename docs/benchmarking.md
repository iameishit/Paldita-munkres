# Benchmarking

```bash
python tools/benchmark.py                    # writes docs/BENCHMARKS.md
python benchmarks/benchmark_dense.py         # square matrices
python benchmarks/benchmark_rectangular.py   # wide and tall matrices
python benchmarks/benchmark_constraints.py   # forbidden cells and gating
python benchmarks/benchmark_memory.py        # peak memory
python benchmarks/benchmark_scaling.py       # growth with n
python benchmarks/benchmark_comparison.py    # against SciPy (needs scipy)
```

Each script prints a table. Timings depend on the machine, so compare ratios, and run on an otherwise idle
machine. The `Benchmarks` workflow runs them all on demand and uploads the output.


## v3 benchmark (`benchmarks/v3_bench.py`)

Both Paldita and SciPy are timed from an in-memory ndarray to `(row_ind, col_ind)`; seeds are fixed; each case reports best-of-7 and the median and p95 of the repeats; the environment is recorded in `benchmarks/results/v3_baseline.json`. The development sandbox has one shared core, so absolute times are noisy and only large ratios should be trusted. Not measured: peak memory, multi-core scaling, cold start, other CPUs/OSes. SciPy's own conversion cost is included for SciPy as it is for Paldita.

Results from the recorded run in `benchmarks/results/v3_baseline.json` (ratio = Paldita time / SciPy time, lower is better for Paldita; single shared core, so treat gaps under about 10% as noise): 5x5 and 10x10 are 3-5x slower (fixed call overhead); 32x32 1.4x slower; 64x64 0.82; 128x128 0.97; 256x256 0.98; 512x512 1.06; small-integer ties 0.69-0.79; rectangular 0.70-0.78; 90% forbidden 0.77 at 64x64 but **1.57 at 256x256 (slower)**; 5,000 batched 8x8 problems 0.56 (about 1.8x faster than a SciPy loop). Exact dynamic repair of a 40x40 problem took a median 1.55 ms against 15.8 ms for a full exact solve. Small cases (n <= 32) carry a problem hash, an agreement check with SciPy and an independent exact certificate (`certificate_verified`).
