# The algorithm

munkres implements the Hungarian method (Kuhn-Munkres) in its *shortest augmenting path* form with dual
potentials.

1. **Warm start.** Each row is reduced by its cheapest allowed cell; rows are greedily matched to free cells
   whose reduced cost is zero. (Only a row reduction is used: a column reduction would not be sound when there are
   more columns than rows.)
2. **Augment.** For each still-unmatched row, grow a tree of reachable columns, always taking the cheapest reduced
   cost. When a free column is reached, the alternating path is flipped, adding one pair. Potentials are adjusted
   by the smallest slack so that all reduced costs stay non-negative.
3. **Finish.** When every row of the smaller side is matched, the matching is optimal: the potentials are a
   feasible dual solution that is tight on every chosen pair (see [`shadow_prices`](analysis.md)).

## Rectangular matrices

The solver always works with at most as many rows as columns. A matrix with more rows is solved as its transpose
and the coordinates are swapped back.

## Forbidden cells

Only allowed cells are stored (as per-row edge lists). If a search finds no reachable free column, the rows visited
can only use the columns visited, one fewer than there are rows: a violation of Hall's marriage condition. That
proof is returned in `UnsolvableMatrix.rows` / `.cols`, so impossible problems are reported immediately.

## Gating

With `max_cost`, pairs costing more are removed and every row gets an extra "leave unmatched" column priced at the
threshold. The result minimises the cost of the pairs made plus the threshold for every unmatched row.

## Complexity

O(n² · m) time for n rows and m columns (n <= m), O(n · m) memory. Each pass of the inner loop visits a new column,
so every call terminates after a bounded number of steps.
