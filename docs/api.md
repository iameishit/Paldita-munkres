# API reference

Everything below is importable from `munkres`. The signatures are generated from the code, so they are always current.

## `DISALLOWED_OBJ`

```python
DISALLOWED_OBJ() -> 'DISALLOWED_OBJ'
```

Sentinel for a forbidden pairing. There is exactly one instance (`DISALLOWED`); it survives `copy`, `deepcopy` and `pickle` (every protocol) as that very same object, so identity checks keep working across processes.


## `Assignment`

```python
Assignment(pairs: 'tuple[tuple[int, int], ...]', total: 'Any', unmatched_rows: 'tuple[int, ...]', unmatched_cols: 'tuple[int, ...]', shape: 'tuple[int, int]', maximize: 'bool' = False, row_labels: 'tuple[Any, ...] | None' = None, col_labels: 'tuple[Any, ...] | None' = None, trace: 'Trace | None' = None) -> None
```

The result of `solve`.

- `Assignment.as_dict(self) -> 'dict[int, int]'`
- `Assignment.labelled(self) -> 'list[tuple[Any, Any]]'`

## `Diagnosis`

```python
Diagnosis(message: 'str', rows: 'tuple[int, ...]', cols: 'tuple[int, ...]') -> None
```

Why a matrix has no complete assignment (a violation of Hall's condition).


## `Munkres`

```python
Munkres() -> 'None'
```

Calculate the Munkres solution to the classical assignment problem.

- `Munkres.compute(self, cost_matrix: 'MatrixLike') -> 'list[tuple[int, int]]'`
- `Munkres.pad_matrix(self, matrix: 'MatrixLike', pad_value: 'Number' = 0) -> 'list[list[Any]]'`

## `Prices`

```python
Prices(row_prices: 'tuple[Any, ...]', col_prices: 'tuple[Any, ...]', total: 'Any') -> None
```

LP dual variables of the assignment problem ("shadow prices").


## `SinkhornResult`

```python
SinkhornResult(plan: 'tuple[tuple[float, ...], ...]', cost: 'float', iterations: 'int', converged: 'bool') -> None
```

Result of `sinkhorn`: the transport `plan`, its cost, and convergence info.


## `Trace`

```python
Trace(shape: 'tuple[int, int]', transposed: 'bool' = False, events: 'list[dict[str, Any]]' = <factory>) -> None
```

The solver's decisions, in order. Obtain one with `solve(matrix, trace=True)`.

- `Trace.steps(self) -> 'list[str]'`
- `Trace.to_html(self) -> 'str'`
- `Trace.to_text(self) -> 'str'`

## `Transport`

```python
Transport(flows: 'dict[tuple[int, int], int]', total_cost: 'Any', shipped: 'int') -> None
```

Result of `transport`: `flows[(supplier, consumer)] = units`.


## `UnsolvableMatrix`

```python
UnsolvableMatrix(message: 'str' = 'Matrix cannot be solved!', rows: 'Sequence[int]' = (), cols: 'Sequence[int]' = ()) -> 'None'
```

Raised when no complete assignment exists.


## `bottleneck`

```python
bottleneck(matrix: 'MatrixLike', *, maximize: 'bool' = False) -> 'Assignment'
```

The assignment whose *single worst pair* is as good as possible (minimise the largest cost; with `maximize=True`, maximise the smallest profit). Among assignments that tie on the worst pair, the lowest total (highest, when maximising) wins. Good for fairness: "nobody gets a terrible job". Raises `UnsolvableMatrix` if no complete assignment exists.

## `build_cost_matrix`

```python
build_cost_matrix(rows: 'Iterable[_A]', cols: 'Iterable[_B]', cost: 'Callable[[_A, _B], Any]') -> 'list[list[Any]]'
```

Build a matrix by calling `cost(row_item, col_item)` for every pair, e.g. a distance between points. If `cost` returns `None` that pairing is `DISALLOWED`.

## `counterfactual`

```python
counterfactual(matrix: 'MatrixLike', row: 'int', col: 'int') -> 'Any'
```

"What would it cost to force `row` onto `col`?" Returns how much the optimal total rises if that pair is made mandatory (0 if some optimal assignment already contains it). Raises `UnsolvableMatrix` if forcing it makes the problem impossible, and `ValueError` for a forbidden or out-of-range cell.

## `diagnose`

```python
diagnose(matrix: 'MatrixLike', *, maximize: 'bool' = False) -> 'Diagnosis | None'
```

Explain why a matrix cannot be solved, without raising.

## `k_best`

```python
k_best(matrix: 'MatrixLike', k: 'int', *, maximize: 'bool' = False) -> 'list[Assignment]'
```

The `k` best complete assignments, best first (Murty's algorithm). Fewer are returned if fewer exist. Each is a distinct set of pairs; ties are ordered arbitrarily.

## `linear_sum_assignment`

```python
linear_sum_assignment(cost_matrix: 'MatrixLike', maximize: 'bool' = False) -> 'tuple[Any, Any]'
```

A drop-in for `scipy.optimize.linear_sum_assignment`: returns `(row_ind, col_ind)`, sorted by row.

## `make_cost_matrix`

```python
make_cost_matrix(profit_matrix: 'MatrixLike', inversion_function: 'Callable[[Any], Any] | None' = None) -> 'list[list[Any]]'
```

Create a cost matrix from a profit matrix by calling `inversion_function()` to invert each value. The inversion function must take one numeric argument (of any type) and return another numeric argument which is presumed to be the cost inverse of the original profit value. If the inversion function is not provided, a given cell's inverted value is calculated as `max(matrix) - value`.

## `print_matrix`

```python
print_matrix(matrix: 'MatrixLike', msg: 'str | None' = None) -> 'None'
```

Convenience function: Displays the contents of a matrix.

## `shadow_prices`

```python
shadow_prices(matrix: 'MatrixLike') -> 'Prices'
```

Solve the problem and return its dual variables (see `Prices`).

## `sinkhorn`

```python
sinkhorn(cost: 'MatrixLike', *, reg: 'float' = 0.1, a: 'Sequence[float] | None' = None, b: 'Sequence[float] | None' = None, max_iter: 'int' = 5000, tol: 'float' = 1e-09) -> 'SinkhornResult'
```

Entropy-regularised optimal transport between two distributions `a` (rows) and `b` (columns), solved with Sinkhorn's algorithm in the log domain (stable for small `reg`). Default marginals are uniform. The smaller `reg`, the closer the plan is to the exact (Hungarian) solution; larger `reg` spreads mass out, giving a *soft*, differentiable matching. `DISALLOWED`/`+inf` cells get zero mass. If the forbidden cells force some cell to *exactly* zero mass, the iteration only approaches that asymptotically and `converged` stays false. Pure Python: meant for small and medium problems.

## `soft_assignment`

```python
soft_assignment(matrix: 'MatrixLike', temperature: 'float' = 0.1, *, max_iter: 'int' = 5000) -> 'list[list[float]]'
```

A *soft* assignment: `result[i][j]` is the share of row `i` given to column `j`. Rows sum to 1 (for a square matrix, columns do too: a doubly stochastic matrix). As `temperature` -> 0 it approaches the Hungarian solution's 0/1 matrix; larger values express uncertainty between near-equal choices. Raises `ValueError` if it does not converge.

## `solve`

```python
solve(matrix: 'MatrixLike', *, maximize: 'bool' = False, max_cost: 'Any' = None, min_profit: 'Any' = None, trace: 'bool' = False) -> 'Assignment'
```

Solve an assignment problem and describe the answer.

## `stable_matching`

```python
stable_matching(proposers: 'Mapping[Any, Sequence[Any]]', receivers: 'Mapping[Any, Sequence[Any]]') -> 'dict[Any, Any]'
```

Stable matching by deferred acceptance (Gale-Shapley), for problems where people have *preferences* instead of costs (students/schools, residents/ hospitals, ...). Each argument maps a name to the partners it finds acceptable, most preferred first. Unacceptable pairings are simply not listed (a pair is only made if *both* list each other).

## `tolerance`

```python
tolerance(matrix: 'MatrixLike', row: 'int', col: 'int') -> 'Any'
```

How much can the cost of a pair in the optimal assignment *rise* before the assignment has to change? (the optimum without that pair, minus the optimum). `float('inf')` if the pair can never be replaced. `ValueError` if the pair is not in the assignment `solve` returns.

## `transport`

```python
transport(supply: 'Sequence[int]', demand: 'Sequence[int]', cost: 'MatrixLike', *, max_units: 'int' = 1000) -> 'Transport'
```

The transportation problem: ship goods from suppliers to consumers at minimum total cost, where `cost[i][j]` is the price per unit from supplier `i` to consumer `j` (`DISALLOWED` = no such route). Ships `min(sum(supply), sum(demand))` units.

## `solve_planned`

`solve_planned(cost, maximize=False, verify=False) -> PlannedResult`. Plans (`plan`), runs the chosen exact
strategy and optionally attaches an independently verified `Certificate`. Raises `UnsolvableMatrix` if infeasible.

## `plan`

`plan(cost, maximize=False) -> Plan`. Deterministic choice among exact strategies (`empty`, `components`,
`native-kernel`, `python-exact`); `Plan.explain()` lists the reasons and the transformation ledger.

## `fingerprint`

`fingerprint(cost) -> Fingerprint`. Structural facts: shape, density, dynamic range, duplicate rows/columns,
components, and exact Monge / inverse-Monge / additive flags.

## `monge_witness`

`monge_witness(cost)` returns `None` iff the matrix is Monge (exact, adjacent 2x2 minors); otherwise `(i, j)`
of a violating 2x2. Matrices with forbidden cells are never reported Monge.

## `inverse_monge_witness`

Same as `monge_witness` for the reversed inequality.

## `is_additive`

True iff `c[i][j] = a[i] + b[j]`; then every assignment is optimal.

## `solve_monge`

`solve_monge(cost, maximize=False) -> (pairs, method)`. O(nm) monotone DP when Monge / inverse Monge is
proved, otherwise the general solver. `method` says which one ran.

## `components`

`components(cost) -> [(rows, cols), ...]` connected components of the allowed-edge graph.

## `solve_by_components`

`solve_by_components(cost, maximize=False) -> (pairs, total, n_components_solved)`. Same semantics as `solve`.

## `topology`

`topology(cost, pairs=None, maximize=False) -> Topology`. Forced / optional / excluded optimal edges, exchange
classes, exact reduced costs and the exact number of optimal assignments (`None` if too large to count).

## `edge_margins`

`edge_margins(cost, maximize=False)`. Exact exclusion and forcing penalty for every allowed cell (re-solves; small problems).

## `certify`

`certify(cost, pairs, maximize=False) -> Certificate | Verification`. Re-derives exact duals independently of any
solver; returns a `Certificate`, or a failed `Verification` (with a negative-cycle witness if suboptimal).

## `verify_pairs`

`verify_pairs(cost, pairs, maximize=False) -> Verification` (truthy iff `pairs` is provably optimal).

## `problem_hash`

`problem_hash(cost, maximize=False) -> str`. Canonical SHA-256 of the exact problem.

## `solve_batch`

`solve_batch(costs, maximize=False, on_infeasible="raise") -> BatchResult`. Many same-shape problems (B, n, m), n <= m, in one native call.

## `rank_maximal`

`rank_maximal(ranks, cost=None) -> RankMaximal`. Lexicographic rank-maximal matching with minimum-cost tie-break.

## `assign_with_inertia`

`assign_with_inertia(cost, previous, switch_cost=0, maximize=False) -> InertiaResult`. Optimal for a per-row switching penalty.

## `AssignmentSolver`

Reusable solver for one fixed shape/dtype (`rows <= cols`); scratch memory is allocated once. `solver.solve(cost)` returns the column of each row.

## `AssignmentState`

Dynamic assignment with exact duals. `update_cost`, `update_costs`, `add_row`, `add_column`, `remove_row`, `remove_column`, then `solve()` repairs only what the change damaged. `verify()` re-proves optimality; `stats` reports rows re-augmented; `snapshot()` / `restore(snap)` keep independent copies of the state.

## `BatchResult`

Result of `solve_batch`: `col_ind[b, i]` and per-problem `status` (0 ok, 1 infeasible, 2 NaN or -inf).

## `Certificate`

Portable optimality proof: pairs, exact duals, objective, `problem_hash`. `verify(cost)` re-checks it; `to_json()` / `from_json()` serialise it.

## `Fingerprint`

Frozen dataclass returned by `fingerprint`.

## `InertiaResult`

Result of `assign_with_inertia`: `pairs`, `cost`, `penalty`, `objective`, `switches`.

## `Plan`

Result of `plan`: `strategy`, `reasons`, `fingerprint`, `ledger`; `explain()` renders them.

## `PlannedResult`

Result of `solve_planned`: `pairs`, `total`, `plan`, optional `certificate`.

## `RankMaximal`

Result of `rank_maximal`: `pairs`, `signature`, `cost`.

## `TemporalAssignment`

Rolling-horizon assignment: `step(cost)` solves each stage optimally given the previous stage (not a global optimum over all stages).

## `Topology`

Result of `topology`: `forced`, `optional`, `excluded_tight`, `exchange_classes`, `reduced_cost`, `n_optimal`, `unique`.

## `Verification`

Truthy/falsy result of a check with `stage` and a precise `reason`.

## `enumerate_optima`

`enumerate_optima(cost, maximize=False, limit=100000)` yields every optimal assignment exactly once (exact; raises `RuntimeError` past `limit`).

## `face_dimension`

`face_dimension(cost, maximize=False)` dimension of the face spanned by all optimal assignments (0 means a unique optimum).

## `canonical_optimum`

`canonical_optimum(cost, maximize=False)` the lexicographically smallest optimal assignment (rows <= cols), found exactly with infinitesimal rational tie-break weights, so it does not depend on solver tie-breaking.

## `most_different_optimum`

`most_different_optimum(cost, reference, maximize=False)` the optimal assignment farthest from `reference` by `matching_distance`.

## `matching_distance`

`matching_distance(a, b)` number of pairs of `a` that are not in `b`.

## `reduce_costs`

`reduce_costs(cost) -> ReducedProblem`. Exact row (rows <= cols) and column (rows >= cols) minimum subtraction with the objective offset recorded; the optimal assignment is unchanged.

## `ReducedProblem`

Result of `reduce_costs`: `matrix`, `row_offsets`, `col_offsets`, `constant` (`original objective = reduced objective + constant`).

## `dulmage_mendelsohn`

`dulmage_mendelsohn(cost) -> DulmageMendelsohn`. Maximum matching size of the allowed-edge graph and the rows / columns some maximum matching leaves unmatched (coarse part only).

## `DulmageMendelsohn`

Result of `dulmage_mendelsohn`: `matching_size`, `rows_missable`, `cols_missable`.

## `exchange_cycles`

`exchange_cycles(cost, maximize=False)` alternating cycles/paths, one per optional unmatched cell; each gives another provably optimal assignment. Covers about 99% of optional cells in testing (rare two-chain moves are not generated).

## `regret_of`

`regret_of(pairs, low, high) -> Regret`. Exact worst-case regret under interval costs, with the worst-case scenario as a checkable certificate.

## `Regret`

Result of `regret_of`: `value`, `scenario`, `optimal_pairs`.

## `minimax_regret_exact`

`minimax_regret_exact(low, high, max_n=7)` assignment with minimum worst-case regret by enumeration (the problem is strongly NP-hard; small n only).

## `count_matchings`

`count_matchings(cost)` exact number of ways to match every row to a distinct allowed column (columns <= 20).

## `gibbs`

`gibbs(cost, beta=1.0) -> Gibbs`. Exact log partition function, edge marginals, expected cost and entropy of the explicit Gibbs model P(M) ~ exp(-beta cost(M)). The marginals are model probabilities, not confidence scores for an optimiser's answer.

## `Gibbs`

Result of `gibbs`: `beta`, `log_z`, `marginals`, `expected_cost`, `entropy`, `n_matchings`.

## `minimize_failing_matrix`

`minimize_failing_matrix(cost, fails)` shrinks a matrix (delete rows/columns, simplify cells) while the predicate `fails` stays true; the result is 1-minimal.

## `maxplus`

Max-plus matrix product: C[i][j] = max_k (A[i][k] + B[k][j]); -inf entries are neutral.

## `minplus`

Min-plus matrix product: C[i][j] = min_k (A[i][k] + B[k][j]); +inf entries are neutral.

## `rank_one_decomposition`

Exact (a, b) with c[i][j] = a[i] * b[j] for a fully finite matrix, or None if the rank exceeds one.

## `solve_line_matching`

Minimum-cost matching of all points x into distinct points y on a line with cost |x - y|**p (p >= 1 integer, convex => order-preserving optimum). No cost matrix is built when len(x) == len(y). Returns `(pairs of original indices, total cost)`.

## `solve_rank_one`

Square rank-one matrices (c = a b^T, any signs): the minimum-sum assignment pairs the smallest a with the largest b (rearrangement inequality). Raises ValueError if the matrix is not square rank-one.

## `tropical_permanent`

Min-plus (or max-plus) permanent over injections of all rows into distinct columns, by subset DP (O(n 2^m m), m <= 20). An independent computation of the assignment optimum; `None` if infeasible.

## `BoundResult`

Result of `solve_with_bound`: `pairs` (None if nothing meets the bound), `total`, `solved` (False when a cheap bound rejected it without solving).

## `prune_edges_by_bound`

Reduced-cost fixing for minimisation: forbid every cell (i, j) with  optimum + reduced_cost(i, j) > upper. Any assignment using such a cell costs more than `upper`, so all assignments with cost <= upper survive. Returns `(pruned matrix, number of cells removed)`.

## `qap_cost`

Koopmans-Beckmann QAP objective: sum_{i,j} flow[i][j] * dist[perm[i]][perm[j]].

## `qap_local_search`

First-improvement 2-swap local search from `perm`. A heuristic: the result is a local optimum, not proven optimal.

## `qap_lower_bound`

Gilmore-Lawler lower bound: L[i][k] = f_ii d_kk + (smallest inner product of row i of F and row k of D off the diagonal, by the rearrangement inequality); the bound is the optimum of the LAP on L. A valid bound for min over permutations of `qap_cost`.

## `solve_with_bound`

Minimisation: an optimal assignment if its cost is <= `upper`, else `pairs=None`. When rows <= cols the sum of row minima is checked first and can reject without solving. (maximize: objective >= `upper`.)

## `Market`

Result of `market_equilibrium`: `assignment`, `prices`, `utilities`, `welfare`.

## `is_competitive_equilibrium`

Independent check: demand is satisfied, unsold items are free, nobody pays more than they value.

## `market_equilibrium`

Welfare-maximising assignment and competitive-equilibrium prices. `DISALLOWED`/None/-inf = not interested.

## `prune_dominated_scenarios`

Indices of scenarios that can matter for a *worst-case (minimax)* objective: a scenario whose every cell is <= another scenario's cell can never be the maximum, so it is dropped (one of equal copies kept).

## `robust_edges`

Cells used by every optimum of every scenario (`always`) and cells optimal in every scenario (`possibly_everywhere`).

## `scenario_assignment`

Assignment minimising the weighted *expected* cost over scenarios (exactly one LAP on the weighted mean).

## `Piece`

One linear piece of the parametric optimum: `start`, `end`, `pairs`, `base`, `slope` (objective = base + slope*lambda).

## `optimality_interval`

The sub-interval of [lo, hi] on which `pairs` is optimal for A + lambda B (None if it never is).

## `parametric_assignment`

Pieces (in order) of the exact optimum envelope on [lo, hi]; cells forbidden in `a` stay forbidden.

## `explain_edge`

Plain-language, exact explanation of why cell (i, j) is / is not in the optimal solution(s).

## `ground_state_marginals`

Fraction of optimal assignments using each cell = the zero-temperature (beta -> infinity) Gibbs marginals.

## `hall_violator`

For rows <= cols: `None` if every row can be matched, else an inclusion-minimal row set S with |N(S)| < |S| (Hall violator) and its neighbourhood N(S).

## `landscape_report`

One call summarising optimum, number of optima, face dimension, feasible-assignment count and Gibbs statistics.

## `near_optimal`

Every assignment whose objective is within `eps` of the optimum (worse direction), as `(gap, pairs)` sorted by gap. Exact: gap = sum of reduced costs of used cells + sum of -v over unused columns, pruned by `eps`. Raises `RuntimeError` past `limit` results.

## `optimum_diameter`

Largest `matching_distance` between two optimal assignments (0 if the optimum is unique).

## `sample_optimum`

Uniformly random optimal assignment (exact: enumerates the optima, so small/medium problems).

## `vertex_indispensability`

Exact objective change (caller's sense: new optimum - old optimum) if a single row / column were deleted from the problem; `None` if the problem becomes infeasible. Re-solves, so small problems.

## `EquivalenceReport`

Result of `check_equivalent`: `agree`, `objectives` per route, `verified`.

## `backends`

Capability descriptors of the execution backends (what each one supports, and its guarantee).

## `check_equivalent`

Solve the same problem with every independent route (default dispatch, pure Python, components, the planner, canonical optimum, tropical subset DP when small) and certify the answer; `agree` is True iff all objectives match.

## `profile_solve`

Stage timeline in nanoseconds (validate, fingerprint, plan, solve, verify) for one planned, verified solve.

## `versions`

Separate version numbers for the package, public API, solver semantics, certificate format and native ABI.

## `Delta`

A set of cell updates `{(row, col): cost}`; `compose(later)` merges two deltas; apply with `AssignmentState.apply`.

## `gibbs_sample`

One exact sample of an assignment from the Gibbs model of `gibbs` (sequential conditional sampling).

## `CertificateCache`

LRU memo of `verify_pairs` keyed by problem hash, pairs and sense; `hits` / `misses` counters.

## `algebraic_optimum`

Optimum of a square matrix with non-negative integer weights, computed only from determinants mod p. Returns None if no perfect matching exists. Error probability <= n/p (the result can only be too large).

## `has_perfect_matching_algebraic`

Randomised existence test for a perfect matching of the allowed cells (see module notes for the error).

## `permanent`

Exact permanent (sum over injections of all rows into distinct columns), subset DP, at most 20 columns.

## `permanent_bounds`

(lower, upper) bounds on the permanent of a square non-negative matrix with a positive diagonal pattern: per(A) = per(D) / (prod r prod c) for the Sinkhorn-scaled doubly stochastic D = R A C, and n!/n^n <= per(D) <= 1. Accurate to the Sinkhorn convergence (floating point), not a proof for rounded input.

## `cooling_schedule`

Geometric inverse-temperature schedule from `start` to `end` (inclusive).

## `energy_barrier`

Smallest `h` such that assignments `a` and `b` are connected by single moves through assignments that are all within `h` of the optimum (the barrier); None if they are not connected within `eps`. Exact.

## `energy_landscape`

Nodes = every assignment within `eps` of optimal (with its gap); edges = single moves between them.

## `k_best_diverse`

Greedy diverse k-best: scan assignments by increasing (gap, pairs) and keep one if it differs from every kept assignment in at least `min_distance` rows. Each pick is the best assignment compatible with the earlier picks (greedy, not a global optimum over sets). Exact enumeration, small problems.

## `temperature_path`

Exact Gibbs statistics (expected cost, entropy, log Z, largest edge marginal) along a list of betas.

## `verify_canonical`

Independent exact proof that `pairs` is the lexicographically smallest optimal assignment: it must be optimal, and for every row, using any smaller allowed column (with the earlier rows fixed) must be infeasible or strictly worse.

## `LazyResult`

Result of `solve_lazy`: `pairs`, `total`, `candidate_cells`, `evaluations`, `rounds`.

## `solve_lazy`

Exact minimum-cost assignment of all n rows (n <= m) with `cost(i, j)` evaluated on demand.

## `solve_points`

Exact minimum-cost matching of points x into points y (rows <= cols) with Euclidean cost (squared by default), using each point's k nearest targets as the initial candidate graph and the exact dual check above.

## `is_popular`

Brute force definition check over *all* matchings (exponential: small instances only, <= 6 applicants).

## `popular_matching`

Return a popular matching `{applicant: post or None}` or None if none exists. Posts are ints >= 0.

## `cvar_assignment`

Assignment minimising CVaR_alpha of the cost over equally likely scenarios: the mean of the worst ceil((1 - alpha) * S) scenario costs. alpha = 0 is the expectation; alpha -> 1 is the worst case.

## `mean_variance_assignment`

Assignment minimising mean + lam * (population) variance of the cost over equally likely scenarios.

## `solve_3d`

Axial 3-index assignment: choose one cell (i, j, k) per i with all j distinct and all k distinct, minimising (or maximising) total cost. NP-hard in general; exact subset DP over (used j, used k), so n <= 8 and dims_j, dims_k <= 10. Forbidden cells: DISALLOWED. Returns `(triples, total)`.

## `ParametricCache`

Memoise `parametric_assignment`; a query whose interval lies inside a cached one is answered by exactly slicing the cached pieces.

## `calibrate_native_threshold`

Time the pure-Python and native solvers on this machine for each matrix side n, label the faster one and learn the crossover cell count with `fit_threshold`. Measurement-driven (hardware-specific); it only chooses between exact solvers.

## `fit_threshold`

Decision stump: the size threshold t minimising misclassifications of 'fastest backend' labels (`low` below t, `high` at or above).

## `parametric_atlas`

Optimal assignment of A + x B1 + y B2 on the grid x in `l1`, y in `l2` (exact solves): region ids per grid point, the assignment of each region and the exact optimum. A *sampled* phase diagram; ties on region boundaries get one arbitrary label.

## `strip_additive`

Square matrices: write c[i][j] = a[i] + b[j] + h[i][j] with h's first row and column zero (exact). Because every row and column is used, opt(c) = sum(a) + sum(b) + opt(h) and the optimal assignments of c and h coincide. `residual` is h.

## `classify_cost`

Descriptive type tags derived from exact facts: dense/sparse, integer/float/rational, square, symmetric, monge, additive, toeplitz, ...

## `hall_violators`

Every inclusion-minimal Hall violator (rows S with |N(S)| < |S|) with its neighbourhood, for rows <= cols and at most 14 rows.

## `solve_group_capacity`

Assignment of all rows (rows <= cols) where at most `caps[g]` columns of group g may be used (a partition-matroid constraint). Exact depth-first search with bounding by the unconstrained optimum; intended for small instances (<= 9 rows).

## `stable_optimum`

Among all optimal assignments, the one whose most fragile cell is the least fragile: maximise the minimum exact exclusion penalty over its cells (a forced cell has infinite margin). Ties -> lexicographically smallest. Returns `(pairs, margin)`.

## `CostSourceRegistry`

Mount named cost sources (matrix, ndarray, or `cost(i, j)` callable with a shape) and solve them by name.

## `Namespace`

Read-only virtual tree over a solved problem: `ls(path)` and `read(path)`; built by `namespace_of`.

## `ProblemTooLargeError`

`ValueError` subclass raised before any allocation when a limit is exceeded; has `.code` (`ROWS_LIMIT`, `COLS_LIMIT`, `CELLS_LIMIT`, `MEMORY_LIMIT`) and `.details`.

## `SolverLimits`

Frozen limits for `solve_limited`: `max_rows`, `max_cols`, `max_cells`, `max_memory_bytes`.

## `choose_representation`

Pick dense / sparse / lazy from shape, density and a memory budget, with the reasons.

## `estimate_density`

Estimated fraction of allowed cells from `samples` random cells, and its standard error (selectivity estimate).

## `estimate_memory`

Rough bytes needed: dense = 8nm; sparse = 24 per allowed cell + 16(n+m); lazy = 16(n+m) plus the candidate graph (assumed 8 per row).

## `latency_profile`

Per-call latency percentiles in nanoseconds (p50, p90, p95, p99, p99.9, max) after a warm-up.

## `namespace_of`

Solve `matrix` with verification and expose everything as a `Namespace`.

## `solve_limited`

`solve_planned` after checking shape-based limits (no allocation happens before the checks).

## `solve_spec`

Declarative interface. `spec` = {"cost": [[number | null]], "objective": "min" | "max", "forbidden": [[i, j], ...], "verify": bool, "limits": {...}}. `null` cells are forbidden. Returns a JSON-safe dict. Raises ValueError with a clear message on bad input.

## `HistoryCache`

Remember solved problems; solve a new same-shape problem by repairing the closest remembered state instead of starting over.

## `compact_log`

Drop cell updates overwritten by a later update of the same cell when no structural change happened in between.

## `fragility_graph`

For each cell of the optimum: the best alternative assignment if that cell were forbidden, and the exact penalty (None if infeasible).

## `next_breakpoint`

First lambda' > lam where the optimal assignment of A + lambda B changes (None if it never changes before `hi`).

## `optimality_length`

Length of the lambda-interval on which `pairs` is optimal for A + lambda B (0 if never, or a single point).

## `parametric_degeneracy`

At every breakpoint: the exact number of optimal assignments there (always >= 2, since two lines meet).

## `parametric_duals`

Exact optimal duals at the midpoint of every piece of the parametric optimum (checkable with `Certificate.verify`).

## `pareto_assignments`

Assignments whose scenario-cost vector is not dominated (no other is <= in every scenario and < in one). Exact enumeration, n <= 8.

## `qap_branch_and_bound`

Exact Koopmans-Beckmann QAP by branch and bound with a Gilmore-Lawler (LAP) bound at every node. Small n (<= ~9).

## `qap_gap`

Gilmore-Lawler bound vs the best 2-swap local-search incumbent: how tight the LAP relaxation is on this instance.

## `scenario_mass`

For every assignment that is optimal in some scenario: the fraction of scenarios in which it is optimal.

## `solve_memmap`

Exact assignment for a matrix stored on disk: rows are streamed from a memory map; only the candidate graph is held in memory.

## `solve_points_grid`

Exact point matching; candidates come from a spatial grid (neighbouring cells, widened until >= k) instead of sorting all targets.

## `var_assignment`

Minimise Value-at-Risk: the smallest B with P(cost <= B) >= 1 - delta over equally likely scenarios (the ceil((1-delta) S)-th smallest cost).

## `ProofGraph`

Hash-chained proof from `solve_with_proof`; replay with `verify_proof`.

## `ProofStep`

One step of a `ProofGraph`: `name`, `input_hash`, `output_hash`, `claim`.

## `crt_objective`

Integer objective of `pairs` reconstructed only from residues modulo several primes (Chinese remainder theorem).

## `rational_reconstruct`

The unique p/q with |p|, q <= sqrt(m/2) and p == a*q (mod m) (Wang's algorithm); ValueError if none.

## `select_backend`

First available backend whose capability descriptor satisfies every need (a boolean flag name or a dtype name); LookupError otherwise.

## `solve_with_proof`

Solve and return the chain: input -> (plan) -> solve -> certify, each step hashed so it can be replayed.

## `validate_passes`

Independently validate every preprocessing pass of this package on `matrix` (negate, transpose, reduce_costs, components).

## `validate_transform`

Exactly check  opt(original) == sign * opt(transformed) + constant  by solving both (the claim a preprocessing pass makes).

## `verify_proof`

Re-derive the problem hash, step-hash chain and certificate from scratch; any tampering fails.

## `QUBO`

Result of `assignment_to_qubo`: `Q`, `offset`, `cells`, `penalty`, `shape` (E(x) = offset + sum Q[a,b] x_a x_b).

## `ExperimentRecorder`

Append-only JSON log of sampler runs; the backend is always labelled classical.

## `anneal_qubo`

Single-flip Metropolis simulated annealing (deterministic for a given seed). A heuristic: no optimality claim.

## `assignment_to_qubo`

Encode min-cost assignment of all rows (rows <= cols) into distinct columns. Rows: P (1 - sum_j x_ij)^2; columns: P x_ij x_i'j for every pair in a column. The default penalty 2 n max|c| + 1 guarantees every QUBO minimiser is a valid optimal assignment.

## `decode_bits`

Selected cells if they form a valid assignment (every row once, columns at most once), else None.

## `qubo_energy`



## `qubo_to_ising`

x = (1 + s) / 2: returns fields h, couplings J (a < b) and constant with E_qubo(x) == const + sum h s + sum J s s.

## `reduce_assignment_problem`

Use the classical optimum's forced cells to shrink the problem before encoding: delete their rows/columns.

## `repair_bits`

Keep the cheapest conflict-free selected cells, then match the remaining rows optimally on the remaining columns.

## `verify_candidate`

Check any candidate assignment (from any sampler) against the exact optimum: feasibility, objective, gap, provably optimal?

## `warm_start_bits`

The exact classical optimum as a bit vector over the QUBO variables (a warm start for any sampler).

## `SEMIRINGS`

Dict of built-in semirings for `semiring_permanent`: `min_plus`, `max_plus`, `counting`, `boolean`, `max_times`.

## `DMBlocks`

Result of `dm_blocks`: `blocks` and `allowed_cells`.

## `Semiring`

`(add, mul, zero, one)` for `semiring_permanent`.

## `dm_blocks`

Strongly connected blocks of the allowed-edge graph under a perfect matching; a cell is in some perfect matching iff its endpoints share a block.

## `is_circulant`

Square and every row is the previous row rotated right by one.

## `is_toeplitz`

c[i][j] == c[i-1][j-1] for all i, j >= 1.

## `is_totally_monotone`

Every 2x2 submatrix (rows i<k, columns j<l) with a[i][j] > a[i][l] also has a[k][j] > a[k][l] (row-minima monotone).

## `is_totally_unimodular`

True iff every entry is in {-1, 0, 1} and every square submatrix has determinant in {-1, 0, 1} (exact; both dims <= max_dim).

## `is_tree_metric`

Metric satisfying the four-point condition: of the three pair sums for any 4 points, the two largest are equal.

## `is_ultrametric`

Symmetric, zero diagonal and d(x,z) <= max(d(x,y), d(y,z)) for all triples.

## `kronecker_factors`

Every exact factorisation C = A (x) B with non-trivial block shapes (A p x q, B (n/p) x (m/q)); B is normalised so its first non-zero is 1... up to a common scalar.

## `leftmost_row_minima`

Column of the leftmost minimum of every row (brute force, O(nm)).

## `monge_violations`

Near-Monge diagnostics: the adjacent 2x2 minors that violate the Monge inequality, with their excess, and the violating fraction.

## `row_minima_monotone`

Leftmost row minima of a totally monotone matrix by divide and conquer (O((n + m) log n) evaluations).

## `semiring_permanent`

Sum (semiring `add`) over injections of all rows into distinct columns of the product (semiring `mul`) of the chosen cells. min_plus = assignment optimum, counting = permanent, boolean = a perfect matching exists. Forbidden cells are skipped; <= 20 columns.

## `tropical_rank`

Largest k such that some k x k minor has a *unique* minimum-cost assignment (min-plus non-singular); at most 6 x 6.

## Constants and type aliases

- `DISALLOWED`
- `DISALLOWED_PRINTVAL`
- `AnyNum`
- `Cell`
- `Matrix`
- `MatrixLike`
- `Number`

## `min_weight_full_bipartite_matching(biadjacency_matrix, maximize=False)`

SciPy-compatible sparse matching (same call as `scipy.sparse.csgraph.min_weight_full_bipartite_matching`).
The stored entries of a sparse biadjacency matrix are the edges and their weights; absent entries are not
edges and explicit zeros are. Returns `(row_ind, col_ind)` sorted by row, covering every vertex of the
smaller side. Raises `ValueError` (`UnsolvableMatrix`) when no full matching exists or a weight is not
finite. Any object with `.tocsr()` is accepted (no SciPy import is needed); a dense array is treated like
`csr_matrix(dense)`. When several optimal matchings exist the returned one may differ from SciPy's, but the
total weight is always the same.

## Production safety: `native_status()`, `reset_native()`, `set_strict_mode()`

The C kernel is a pure accelerator, and munkres is built so that it can never take your service down.
If the extension is missing, was built for a different ABI, fails its load-time self-test, or raises or
misbehaves at run time, the call is answered by the exact pure-Python solver instead (same optimal
objective), the failure is logged once per call site on the `munkres` logger, and after 3 unexpected
failures the kernel is switched off for the process (circuit breaker). Ordinary problems with the data
(an infeasible matrix, NaN) are still reported as errors, never mistaken for kernel faults.

- `native_status()` returns a JSON-friendly dict (`available`, `version`, `disabled_reason`, `failures`,
  `recent_failures`, `breaker_limit`, `strict`) for health checks and logs. It never raises.
- `reset_native()` clears the counters and the breaker, reloads the kernel and returns whether it is active.
- `set_strict_mode(strict=True)` makes unexpected native failures raise instead of falling back (use it in
  CI; the environment variable `PALDITA_STRICT=1` does the same). `PALDITA_NO_NATIVE=1` turns the kernel off.

## `native_status`

```python
native_status() -> 'dict[str, Any]'
```

Health of the native accelerator for logs and health checks: `available`, `version`, `disabled_reason`, `failures` (unexpected native errors absorbed so far), `recent_failures` (the last five), `breaker_limit` and `strict`. Safe to call at any time and never raises. See [production safety](production.md).

## `reset_native`

```python
reset_native() -> 'bool'
```

Clears the failure counters and the circuit breaker, reloads the native kernel (re-running its load-time self-test) and returns whether it is active. Use after fixing the cause of a disabled kernel.

## `set_strict_mode`

```python
set_strict_mode(strict: 'bool' = True) -> 'None'
```

In strict mode an unexpected native failure raises instead of falling back to pure Python. For CI and tests, not production. The environment variable `PALDITA_STRICT=1` does the same.
