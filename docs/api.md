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

## Constants and type aliases

- `DISALLOWED`
- `DISALLOWED_PRINTVAL`
- `AnyNum`
- `Cell`
- `Matrix`
- `MatrixLike`
- `Number`
