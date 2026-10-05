# Analysis tools

```python
from munkres import bottleneck, counterfactual, k_best, shadow_prices, tolerance
m = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
```

| Function | Answers |
|----------|---------|
| `shadow_prices(m)` | The LP dual variables. `cost[i][j] >= row_price[i] + col_price[j]` for every allowed cell, with equality on the chosen pairs, and the prices sum to the optimal total. Anyone can re-check this in O(rows * columns): a certificate of optimality. |
| `counterfactual(m, i, j)` | How much the optimal total rises if pair `(i, j)` is made mandatory. |
| `tolerance(m, i, j)` | How much a chosen pair's cost may rise before the assignment must change (`inf` if it never must). |
| `k_best(m, k)` | The `k` best assignments, best first (Murty's algorithm); `maximize=True` supported. |
| `bottleneck(m)` | The assignment whose single worst pair is as good as possible; ties broken by total. |
