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
