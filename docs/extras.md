# Related problems

| Function | Problem |
|----------|---------|
| `stable_matching(proposers, receivers)` | Stable matching from preference lists (Gale-Shapley). Proposer-optimal; swap the arguments to favour the other side. |
| `transport(supply, demand, cost)` | The transportation problem: ship units from suppliers to consumers at least cost. Solved exactly by treating each unit as its own row/column, so it suits modest quantities (`max_units`, default 1000). |
| `sinkhorn(cost, reg=0.1, a=None, b=None)` | Entropy-regularised optimal transport between two distributions. Smaller `reg` approaches the exact solution; larger values spread mass out. |
| `soft_assignment(matrix, temperature)` | A row-stochastic "soft" assignment from Sinkhorn. As the temperature falls it approaches the hard 0/1 assignment. |

Forbidden cells get zero mass in Sinkhorn. If forbidden cells force some cell to exactly zero mass, the iteration
approaches that only gradually and reports `converged=False`; raise `reg` or `max_iter`.
