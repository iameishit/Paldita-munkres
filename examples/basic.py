"""The textbook example: three workers, three jobs."""

from munkres import Munkres, solve

cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]

print("pairs:", Munkres().compute(cost))
result = solve(cost)
print("total cost:", result.total)
for worker, job in result:
    print(f"worker {worker} -> job {job} (cost {cost[worker][job]})")
