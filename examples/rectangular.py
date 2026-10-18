"""More jobs than workers: some jobs stay unmatched."""

from munkres import solve

cost = [[4, 1, 3], [2, 0, 5]]
result = solve(cost)
print("pairs:", result.pairs)
print("unmatched jobs:", result.unmatched_cols)
print(
    "tall version, unmatched workers:",
    solve([list(c) for c in zip(*cost, strict=True)]).unmatched_rows,
)
