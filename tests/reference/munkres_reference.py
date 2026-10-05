"""
An independent reference solver for the assignment problem, written for obviousness, not speed:
dynamic programming over subsets of columns (exact for ints, Fractions and Decimals). It shares no
code with the library and is used as an oracle on matrices with up to 16 columns on the smaller side.
"""

from munkres import DISALLOWED

_INF = float("inf")


def min_cost_assignment(matrix):
    """Return `(cost, pairs)` of an optimal complete assignment, or `None` if none exists."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return 0, []
    flip = rows > cols
    grid = [list(column) for column in zip(*matrix, strict=True)] if flip else matrix
    n, m = len(grid), len(grid[0])
    if m > 16:
        raise ValueError("reference solver is limited to 16 columns on the smaller side")
    best = {0: (0, ())}
    for r in range(n):
        following = {}
        for mask, (cost, pairs) in best.items():
            for c in range(m):
                value = grid[r][c]
                if mask >> c & 1 or value is DISALLOWED or value == _INF:
                    continue
                new_mask, new_cost = mask | 1 << c, cost + value
                if new_mask not in following or new_cost < following[new_mask][0]:
                    following[new_mask] = (new_cost, (*pairs, (r, c)))
        best = following
        if not best:
            return None
    cost, pairs = min(best.values(), key=lambda entry: entry[0])
    return cost, sorted((c, r) if flip else (r, c) for r, c in pairs)
