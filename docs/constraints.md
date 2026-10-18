# Constraints

## Forbidden pairs

```python
from munkres import DISALLOWED as D, solve
print(solve([[4, D, 3], [2, 0, D], [3, 2, 2]]).pairs)   # ((0, 2), (1, 1), (2, 0))
```

`float("inf")` means the same. If the forbidden cells make a complete assignment impossible,
`UnsolvableMatrix` is raised at once. Its `.rows` and `.cols` name rows that can only use fewer columns than there
are rows. `diagnose(matrix)` returns the same explanation without raising.

## Gating

```python
solve(cost, max_cost=3)                    # never pair above 3; leave rows unmatched instead
solve(profit, maximize=True, min_profit=6) # never pair below 6
```

Gated problems can never be unsolvable. A pair costing exactly the threshold is a tie with leaving the row
unmatched. `max_cost` is for minimising and `min_profit` for maximising; using the wrong one raises `ValueError`.

## Forcing and forbidding pairs after the fact

`counterfactual(matrix, row, col)` tells you what forcing a pair would cost; `k_best` lists runners-up. See
[analysis](analysis.md).
