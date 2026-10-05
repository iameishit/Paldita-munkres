# Release notes: 2.0.0rc1

Faster, typed, thread-safe, and impossible matrices now raise a clear error immediately. The full notes are in
[V2_RELEASE.txt](../V2_RELEASE.txt); the complete list of changes is in [CHANGELOG.md](../CHANGELOG.md).

## Highlights

- 5x to 50x faster than 1.1.4 on random matrices (on par when almost all costs tie)
- `solve()` with gating, maximise, labels and traces; `diagnose()`; SciPy-compatible `linear_sum_assignment()`
- Optimality certificates, what-if analysis, k-best, bottleneck, stable matching, transportation, Sinkhorn
- A `munkres` command line tool
- Python 3.10 to 3.14, fully typed, no runtime dependencies

## Upgrading

Ragged matrices, `NaN` and `-inf` now raise `ValueError`; `+inf` means forbidden; Python 3.10+ is required.
See [docs/migration.md](../docs/migration.md).
