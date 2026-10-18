"""Several features working together, and the installed command."""

import json
import subprocess
import sys

from munkres import (
    Munkres,
    bottleneck,
    build_cost_matrix,
    counterfactual,
    k_best,
    shadow_prices,
    solve,
    transport,
)

from ..reference.munkres_reference import min_cost_assignment


def test_build_solve_certify_and_explore():
    workers = [(0, 0), (4, 1), (2, 5), (7, 7)]
    jobs = [(1, 1), (5, 0), (2, 4), (8, 6)]
    cost = build_cost_matrix(workers, jobs, lambda w, j: abs(w[0] - j[0]) + abs(w[1] - j[1]))
    answer = solve(cost)
    assert answer.total == min_cost_assignment(cost)[0]
    assert shadow_prices(cost).total == answer.total  # the dual certificate agrees
    assert k_best(cost, 1)[0].total == answer.total
    assert all(counterfactual(cost, r, c) == 0 for r, c in answer.pairs)
    assert bottleneck(cost).total >= answer.total  # min-sum is a lower bound on any assignment
    assert list(answer) == Munkres().compute(cost)


def test_unit_transport_equals_assignment():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    plan = transport([1, 1, 1], [1, 1, 1], cost)
    assert plan.total_cost == solve(cost).total and plan.shipped == 3


def test_command_line_agrees_with_the_library(fixture_file):
    path = fixture_file("classic_3x3.csv")
    done = subprocess.run(
        [sys.executable, "-m", "munkres", path, "--json"],
        capture_output=True, text=True, timeout=60, check=True,
    )  # fmt: skip
    payload = json.loads(done.stdout)
    from munkres.cli import parse_matrix

    with open(path, encoding="utf-8") as handle:
        matrix = parse_matrix(handle.read())
    assert payload["total"] == solve(matrix).total
    assert [tuple(p) for p in payload["pairs"]] == list(solve(matrix).pairs)
