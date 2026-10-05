"""Certificates, what-ifs, runners-up and the fairest assignment."""

from munkres import bottleneck, counterfactual, k_best, shadow_prices, solve, tolerance

cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
print("optimal total:", solve(cost).total)
print("dual prices (a proof of optimality):", shadow_prices(cost))
print("cost of forcing worker 0 onto job 0:", counterfactual(cost, 0, 0))
print("how far (0, 1) can get dearer:", tolerance(cost, 0, 1))
print("three best totals:", [a.total for a in k_best(cost, 3)])
print("fairest (smallest worst pair):", bottleneck(cost).pairs)
