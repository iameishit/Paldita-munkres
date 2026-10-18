# Getting started

You have workers and jobs, and a cost for every pairing. Find the cheapest one-to-one assignment.

```python
from munkres import Munkres, solve

cost = [[4, 1, 3],
        [2, 0, 5],
        [3, 2, 2]]

print(Munkres().compute(cost))     # [(0, 1), (1, 0), (2, 2)]
result = solve(cost)
print(result.total)                # 5
```

- `Munkres().compute(matrix)` returns `(row, column)` pairs.
- `solve(matrix)` returns an [`Assignment`](api.md) with the total cost, unmatched rows and columns and, for a
  pandas DataFrame, your labels.

## More rows than columns (or the reverse)

`min(rows, columns)` pairs are returned; the extras are reported as `unmatched_rows` / `unmatched_cols`.

## Forbidding pairings

Use `DISALLOWED` (or `float("inf")`) for pairs that must not be used. See [constraints](constraints.md).

## Maximising

`solve(profit, maximize=True)`.

## Next

[Analysis tools](analysis.md), [related problems](extras.md), the [command line](api.md), [FAQ](faq.md).
