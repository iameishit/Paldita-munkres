"""Forbid pairings, and find out why a problem has no solution."""

from munkres import DISALLOWED as D
from munkres import UnsolvableMatrix, diagnose, solve

print(solve([[4, D, 3], [2, 0, D], [3, 2, 2]]).pairs)

impossible = [[1, D, D], [1, D, D], [1, 2, 3]]
try:
    solve(impossible)
except UnsolvableMatrix as error:
    print("error:", error)
print("diagnosis:", diagnose(impossible))
print("with gating it can always be answered:", solve(impossible, max_cost=10).pairs)
