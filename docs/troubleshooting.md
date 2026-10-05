# Troubleshooting

**`UnsolvableMatrix: No complete assignment exists: rows [0, 1] can only be matched to columns [0] ...`**
Forbidden cells leave too few columns for those rows. Allow more cells, or use gating
(`solve(m, max_cost=...)`) to leave some rows unmatched. `diagnose(m)` returns the explanation without raising.

**`ValueError: cost matrix is not rectangular: row 1 has 1 entries but row 0 has 2`**
Every row needs the same length. Mark missing pairings with `DISALLOWED`.

**`ValueError: cell [i][j] is NaN` / `-infinity`**
These have no meaning as costs. Use `DISALLOWED` (or `+inf`) for forbidden pairings.

**`ValueError: ... +infinity ...` with `maximize=True`**
Use `-inf` or `DISALLOWED` to forbid a pairing when maximising.

**`TypeError: cell [i][j] is not a number`**
Convert strings or `None` before solving; `build_cost_matrix` turns a `None` return into `DISALLOWED`.

**`transport` raises `max_units`**
Each unit becomes its own row/column. Reduce the quantities or raise `max_units`.

**`sinkhorn` reports `converged=False`**
Raise `reg` (or `temperature`), or `max_iter`. Forbidden cells that force exact zeros converge slowly.

**A different answer than SciPy**
When several assignments are equally cheap either may be returned; compare the totals.

**`munkres: command not found`**
Install into the active environment (`pip install munkres`) or use `python -m munkres`.

**`ModuleNotFoundError: pandas` / `numpy`**
They are optional. Install them only if you pass arrays or DataFrames.
