<p align="center">
  <img src="https://raw.githubusercontent.com/iameishit/Paldita-munkres/main/assets/banner.jpg" alt="Paldita Munkres V2: Faster. Smarter. More Robust." width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/iameishit/Paldita-munkres/main/assets/logo.png" alt="Paldita Munkres V2 badge" width="150">
</p>

# munkres

[![CI](https://github.com/iameishit/Paldita-munkres/actions/workflows/ci.yml/badge.svg)](https://github.com/iameishit/Paldita-munkres/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE.md)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-3776AB)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![Typed](https://img.shields.io/badge/typing-strict-informational)
![Release](https://img.shields.io/badge/release-v2.0.0-2f6fed)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

The **Munkres (Hungarian) algorithm** for the *assignment problem*: given a cost
for every (worker, job) pair, find the one-to-one assignment with the lowest
total cost. Pure Python, no dependencies, fully typed, thread-safe.

```text
            Job A  Job B  Job C
Worker 0      4      1      3
Worker 1      2      0      5        ->   0->B (1)   1->A (2)   2->C (2)   total 5
Worker 2      3      2      2
```

## Install

```bash
pip install munkres==2.0.0
pip install git+https://github.com/iameishit/Paldita-munkres
```

Needs Python 3.10+. numpy and pandas are **optional**: if you have them, arrays
and DataFrames are accepted as input.

## Quick start

```python
from munkres import Munkres

cost = [[4, 1, 3],
        [2, 0, 5],
        [3, 2, 2]]

pairs = Munkres().compute(cost)
print(pairs)                                   # [(0, 1), (1, 0), (2, 2)]
print(sum(cost[r][c] for r, c in pairs))       # 5
```

`compute` returns `(row, column)` pairs sorted by row. Rectangular matrices
work too: `min(rows, columns)` pairs are returned.

### Forbidding a pairing

```python
from munkres import DISALLOWED, Munkres

cost = [[4, DISALLOWED, 3],
        [2, 0, DISALLOWED],
        [3, 2, 2]]
print(Munkres().compute(cost))                 # [(0, 2), (1, 1), (2, 0)]
```

`float("inf")` means the same as `DISALLOWED`.

### Maximising profit instead of minimising cost

```python
from munkres import Munkres, make_cost_matrix

profit = [[5, 9, 1], [10, 3, 2], [8, 7, 4]]
pairs = Munkres().compute(make_cost_matrix(profit))
print(sum(profit[r][c] for r, c in pairs))     # 23
```

### When there is no valid assignment

Impossible constraints raise immediately (they used to hang forever in 1.x), and
the exception tells you *why*:

```python
from munkres import DISALLOWED as D, Munkres, UnsolvableMatrix

try:
    Munkres().compute([[1, D, D], [1, D, D], [1, 2, 3]])
except UnsolvableMatrix as error:
    print(error)       # No complete assignment exists: rows [0, 1] can only be matched to columns [0] ...
    print(error.rows, error.cols)              # (0, 1) (0,)
```

### numpy and pandas

```python
import numpy as np
from munkres import Munkres

print(Munkres().compute(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])))
```

Inputs are never modified. `Fraction` and `Decimal` costs work as well.

## More than `compute()`

### A richer result: `solve`

```python
from munkres import solve

result = solve([[4, 1, 3], [2, 0, 5]])         # 2 workers, 3 jobs
print(result.pairs)                            # ((0, 1), (1, 0))
print(result.total)                            # 3
print(result.unmatched_cols)                   # (2,)
```

`maximize=True` maximises (exactly, even for `Decimal`/`Fraction`).
**Gating** (`max_cost=`, or `min_profit=` when maximising) refuses bad pairs and
leaves rows unmatched instead, which is what object tracking needs:

```python
from munkres import solve

cost = [[1, 9, 9], [9, 2, 9], [9, 9, 8]]
print(solve(cost, max_cost=3).pairs)           # ((0, 0), (1, 1))
profit = [[5, 9, 1], [10, 3, 2], [8, 7, 4]]
print(solve(profit, maximize=True).total)      # 23
```

With a pandas DataFrame you get your labels back:

```python
import pandas as pd
from munkres import solve

frame = pd.DataFrame([[4, 1], [2, 0]], index=["w1", "w2"], columns=["jobA", "jobB"])
print(solve(frame).labelled())                 # [('w1', 'jobB'), ('w2', 'jobA')]
```

### Why is there no answer? `diagnose`

```python
from munkres import DISALLOWED as D, diagnose

problem = diagnose([[1, D, D], [1, D, D], [1, 2, 3]])
print(problem.rows, problem.cols)              # (0, 1) (0,)
```

### Watch it think: `trace`

```python
from munkres import solve

trace = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True).trace
print(trace.to_text().splitlines()[0])         # Hungarian algorithm on a 3x3 matrix (10 steps)
```

`trace.to_html()` gives a static page you can open or share.

### Drop-in for SciPy, and a matrix builder

```python
from munkres import build_cost_matrix, linear_sum_assignment, solve

rows, cols = linear_sum_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
print(rows, cols)                              # [0, 1, 2] [1, 0, 2]
workers, jobs = [(0, 0), (5, 5)], [(1, 1), (6, 4)]
cost = build_cost_matrix(workers, jobs, lambda w, j: abs(w[0] - j[0]) + abs(w[1] - j[1]))
print(solve(cost).pairs)                       # ((0, 0), (1, 1))
```

## Analysing a problem

```python
from munkres import bottleneck, counterfactual, k_best, shadow_prices, tolerance

m = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
print(shadow_prices(m).total)                  # 5
print(counterfactual(m, 0, 0))                 # 1
print(tolerance(m, 0, 1))                      # 1
print([a.total for a in k_best(m, 3)])         # [5, 6, 6]
print(bottleneck([[1, 5], [5, 9]]).pairs)      # ((0, 1), (1, 0))
```

- `shadow_prices`: the LP duals, a certificate anyone can re-check that the answer is optimal.
- `counterfactual(m, i, j)`: what forcing a pair would cost. `tolerance(m, i, j)`: how much a chosen pair's cost may rise before the answer changes.
- `k_best(m, k)`: the k best assignments (Murty's algorithm).
- `bottleneck(m)`: make the *worst single pair* as good as possible.

## Related problems

```python
from munkres import soft_assignment, stable_matching, transport

men = {"A": ["x", "y", "z"], "B": ["y", "x", "z"], "C": ["x", "y", "z"]}
women = {"x": ["B", "A", "C"], "y": ["C", "A", "B"], "z": ["A", "B", "C"]}
print(stable_matching(men, women))             # {'A': 'z', 'B': 'x', 'C': 'y'}
plan = transport([3, 2], [2, 3], [[1, 5], [4, 2]])
print(plan.flows)                              # {(0, 0): 2, (0, 1): 1, (1, 1): 2}
print(plan.total_cost)                         # 11
print(round(soft_assignment([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 0.05)[0][0], 3))  # 1.0
```

- `stable_matching`: preferences instead of costs (Gale-Shapley, proposer-optimal).
- `transport`: ship quantities from suppliers to consumers at least cost.
- `sinkhorn` / `soft_assignment`: entropic optimal transport, a soft, differentiable matching.

## Command line

```bash
munkres costs.csv --maximize --json
printf '4,1,3\n2,0,5\n3,2,2\n' | munkres - --trace
```

Matrix files are comma- or space-separated; an empty cell, `D`, `x` or `inf` forbids a pairing.
`python -m munkres` with no arguments runs a built-in self-check.

## Input rules

| input | result |
|-------|--------|
| ragged rows | `ValueError` |
| `NaN`, `-inf` | `ValueError` |
| a non-number cell | `TypeError` |
| empty matrix (no rows or no columns) | `[]` |
| `DISALLOWED`, `+inf` | pairing forbidden |
| impossible constraints | `UnsolvableMatrix` (a `ValueError`) with a Hall's-theorem witness |

## Performance

Pure-Python O(n²·m). Measured on one machine: 200x200 in ~0.1 s, 500x500 in ~0.8 s, 1000x1000 in ~4 s
(about 5-50x faster than munkres 1.1.4 on random matrices, and on par when almost all costs tie; numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), reproducible with
`python tools/benchmark.py`).

<p align="center">
  <img src="https://raw.githubusercontent.com/iameishit/Paldita-munkres/main/assets/benchmark.png" alt="Benchmark results: munkres 1.1.4 vs munkres 2.x vs SciPy, solve time on a log scale" width="100%">
</p>

For very large or latency-critical problems use a compiled solver such as
`scipy.optimize.linear_sum_assignment`; this library's niche is zero dependencies, `DISALLOWED` support, exact
`Decimal`/`Fraction` arithmetic and clear errors. `munkres.linear_sum_assignment` has the same call signature, so
switching either way is a one-line change.

## Migrating from 1.x

2.0 keeps the `Munkres().compute()`, `make_cost_matrix`, `print_matrix`,
`DISALLOWED` and `UnsolvableMatrix` API, and `Munkres().compute()` is **not** deprecated. What changed:

- ragged matrices now raise `ValueError` (they used to be silently mis-solved);
- `NaN` / `-inf` raise `ValueError` (they used to hang), `+inf` = `DISALLOWED`;
- impossible matrices raise `UnsolvableMatrix` (they used to hang);
- Python 3.10+ only; `setup.py` is gone (`pyproject.toml`).

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## What's new in 2

Faster, typed, thread-safe, and impossible matrices now raise a clear error immediately. New: `solve()`
with gating, labels and traces, optimality certificates, k-best, bottleneck, stable matching, transportation,
Sinkhorn and a `munkres` command. Full details: [V2_RELEASE.txt](V2_RELEASE.txt) and
[CHANGELOG.md](CHANGELOG.md). Documentation: [docs/index.md](docs/index.md).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `pip install -e .` and
`tools/audit.sh`.

## Credits and license

Created by [Brian M. Clapper](https://software.clapper.org/munkres/) (2008-2020);
maintained and extended since 2026 by Eishit Nigam. Licensed under the
[Apache License 2.0](LICENSE.md); see [NOTICE](NOTICE).
