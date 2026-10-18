"""Maximise profit instead of minimising cost, and refuse poor pairings."""

from munkres import solve

profit = [[5, 9, 1], [10, 3, 2], [8, 7, 4]]
print("best total profit:", solve(profit, maximize=True).total)
print("only pairs earning at least 8:", solve(profit, maximize=True, min_profit=8).pairs)
