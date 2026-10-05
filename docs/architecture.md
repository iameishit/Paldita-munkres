# Architecture

```text
src/munkres/
  __init__.py   public API and __version__
  _core.py      validation, DISALLOWED, the solver (_solve/_assign), Munkres, make_cost_matrix, print_matrix
  _api.py       solve(), Assignment, diagnose(), linear_sum_assignment(), build_cost_matrix()
  _analysis.py  shadow_prices, counterfactual, tolerance, k_best, bottleneck
  _extras.py    stable_matching, transport, sinkhorn, soft_assignment
  _trace.py     Trace (step-by-step record, text and HTML)
  cli.py        the `munkres` command
  __main__.py   python -m munkres
```

Data flow: any input (lists, tuples, numpy, DataFrame) -> `_as_lists` (a private copy) -> `_validated_edges`
(types, NaN/inf, shape; builds per-row allowed edges) -> `_assign` (orientation, optional gating) -> `_solve` (the
solver) -> pairs -> `Assignment`.

Design rules: no runtime dependencies; no state on objects (thread-safe); the caller's data is never modified;
infinity is only compared, never used in arithmetic (so `Decimal` and `Fraction` work); every loop is bounded.
