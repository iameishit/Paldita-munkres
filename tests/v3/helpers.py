"""Brute-force oracles shared by the v3 tests (independent of every solver in the package)."""

import itertools

from munkres import DISALLOWED


def all_optima(a, maximize=False):
    """(best_value, [frozenset(pairs), ...]) over all min(n, m)-cardinality assignments; None if infeasible."""
    n, m = len(a), len(a[0])
    sols = []
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            if any(a[i][c] is DISALLOWED for i, c in enumerate(cols)):
                continue
            sols.append((sum(a[i][c] for i, c in enumerate(cols)), frozenset(enumerate(cols))))
    else:
        for rows in itertools.permutations(range(n), m):
            if any(a[r][j] is DISALLOWED for j, r in enumerate(rows)):
                continue
            sols.append(
                (
                    sum(a[r][j] for j, r in enumerate(rows)),
                    frozenset((r, j) for j, r in enumerate(rows)),
                )
            )
    if not sols:
        return None
    best = (max if maximize else min)(s[0] for s in sols)
    return best, [s[1] for s in sols if s[0] == best]
