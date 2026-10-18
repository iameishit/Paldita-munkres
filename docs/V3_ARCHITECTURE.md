# v3 architecture

This documents what exists in the repository, with evidence, and what does not.

## Language and build: C as an optional CPython extension

Decision: a small C (C11) kernel built as the extension module `munkres._lapc`, with the pure-Python solver as the automatic fallback.

Why: it was the only compiled toolchain available in the development environment (gcc; no Rust, Zig or clang), so the choice is
constrained, not benchmarked against alternatives. Rust was **not** evaluated. What the evidence does support:

- `ctypes` was tried first and rejected: the project's own security tests forbid importing it, and the call overhead was about 14 us.
  The extension module uses the buffer protocol (no NumPy needed at build time), checks buffer sizes, releases the GIL, and cut small-matrix call overhead to about 3 us.
- Zero runtime dependencies are kept (`dependencies = []`). NumPy is only used if the caller passes arrays.
- Safety boundary: all native code is in `src/munkres/_native/`; Python validates shapes and dtypes, the kernel rejects NaN and -inf and refuses inputs whose duals could overflow (it returns a code and Python re-solves exactly).
- Packaging: configured in `pyproject.toml` (`ext-modules`, `optional = true`), so a missing compiler leaves a working pure-Python install. The pyproject build is verified in CI: the release workflow builds platform wheels with cibuildwheel and tests each one (too old); the equivalent compile was verified with a temporary `setup.py`. No wheel or sdist install was tested. Compiler flags are portable (`-O3 -fno-trapping-math -fno-math-errno -funroll-loops`; no `-march=native`, no `-ffast-math`).

## Layers that exist

```
Python API (solve, linear_sum_assignment, ...)           src/munkres/_api.py, _dense.py
  -> validation + exact/float domain check                src/munkres/_core.py
  -> planner (facts -> strategy + reasons + ledger)       src/munkres/_plan.py, _structure.py (fingerprint)
  -> solvers: native C kernel | pure-Python | components  src/munkres/_native/, _core.py, _components.py
  -> independent verifier + portable certificates         src/munkres/_verify.py
  -> analysis: topology, optima, regret, Gibbs, dynamic   _topology.py, _landscape.py, _robust.py, _gibbs.py, _dynamic.py
```

The planner is deterministic (no learned model): empty, components (more than one non-trivial connected component), native kernel, pure Python. It only chooses among exact algorithms and can attach an independently verified `Certificate` (`verify=True`). Monge structure is detected and reported but not used for speed: measured at n=100..400, the exact Monge DP is 12-25x slower than the native kernel.

## Mathematical contract (summary; see `numerical-contract.md`)

For n rows and m columns the solvers assign min(n, m) rows/columns at minimum total cost over allowed cells; forbidden cells are `DISALLOWED` or +inf; NaN is rejected; `maximize` is solved exactly as minimisation of the negation. Verification: primal feasibility, dual feasibility, tightness on matched cells, zero duals on unused columns and equal primal/dual objectives, all on exact rationals, with duals re-derived by Bellman-Ford independently of any solver.

## Public additions in v3 (all tested against brute force or exact oracles)

Verification and certificates; structure (`fingerprint`, Monge witnesses, additive test, `solve_monge`); `components`; `topology` (forced / optional / excluded edges, exchange classes, exact number of optima), `edge_margins`, `enumerate_optima`, `face_dimension`, `canonical_optimum`, `exchange_cycles`, `reduce_costs`, `dulmage_mendelsohn`; `AssignmentState` (dynamic repair with exact duals, snapshots); `assign_with_inertia` and `TemporalAssignment`; `rank_maximal`; `regret_of` / `minimax_regret_exact`; `gibbs` / `count_matchings`; `solve_batch`; `AssignmentSolver`; `minimize_failing_matrix`; CLI `doctor`, `inspect`, `certify`, `verify`.

Third batch: algebraic (determinant-based) matching and permanent tools (`has_perfect_matching_algebraic`, `algebraic_optimum`, `permanent`, `permanent_bounds`); `popular_matching`; energy landscape and barriers, diverse k-best, temperature paths, `verify_canonical`; `cvar_assignment`, `mean_variance_assignment`, `solve_3d`; lazy exact costs (`solve_lazy`, `solve_points`).

Second batch (also brute-force or exact-oracle tested): sensitivity and landscape (`hall_violator`, `optimum_diameter`, `vertex_indispensability`, `explain_edge`, `near_optimal`, `sample_optimum`, `ground_state_marginals`, `landscape_report`); parametric assignment (`parametric_assignment`, `optimality_interval`); tropical and closed-form tools (`minplus`, `maxplus`, `tropical_permanent`, `solve_rank_one`, `solve_line_matching`); markets and scenarios (`market_equilibrium`, `is_competitive_equilibrium`, `scenario_assignment`, `prune_dominated_scenarios`, `robust_edges`); LAP-as-a-component (`solve_with_bound`, `prune_edges_by_bound`, `qap_lower_bound`, `qap_local_search`); exact `gibbs_sample`; verification and replay tooling (`CertificateCache`, `check_equivalent`, `profile_solve`, `backends`, `versions`, `Delta`, event log `export_log` / `from_log`).

## Not built in 3.0

Rust or GPU backends, SIMD/hardware-specific kernels, multithreading (measurements were not taken on multi-core hardware, sored), auction / cost-scaling / sparse solvers, geometric and lazy-cost solvers, quantum / Ising / FPGA / algebraic-NC backends, learned planning, QAP relaxations, online assignment. Temporal assignment is a rolling per-stage policy, not a global optimum over all stages. `topology`, `enumerate_optima`, `regret`, `gibbs` and the dynamic state use exact rational arithmetic and are intended for small and medium problems.
