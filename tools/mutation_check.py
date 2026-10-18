#!/usr/bin/env python
"""
Lightweight mutation testing for the solver.

Each mutant is ONE deliberate bug (flip a comparison, swap a sign, drop a
guard...). The test-suite must FAIL on every mutant; a mutant that survives
means a real bug of that shape could slip through unnoticed.

    python tools/mutation_check.py munkres/_core.py [--only LABEL ...]

Runs against a scratch copy of the repo; your working tree is never modified.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

# (label, exact text to replace, replacement). Text must occur exactly once.
MUTANTS = [
    ("tie-break <  -> <=  (augment search)", "if cur < minv[j]:", "if cur <= minv[j]:"),
    ("delta pick   <  -> <=", "if mv < delta:", "if mv <= delta:"),
    ("row potential += -> -=", "u[p[j]] += delta", "u[p[j]] -= delta"),
    ("col potential -= -> +=", "v[j] -= delta", "v[j] += delta"),
    ("minv update - -> +", "minv[j] = mv - delta", "minv[j] = mv + delta"),
    ("minv inf guard inverted", "if mv is not inf:", "if mv is inf:"),
    ("reduced cost drops v[j]", "cur = cost - ui - v[j]", "cur = cost - ui"),
    ("reduced cost drops u[i0]", "cur = cost - ui - v[j]", "cur = cost - v[j]"),
    (
        "warm start: wrong tightness test",
        "p[j] == -1 and cost == low",
        "p[j] == -1 and cost != low",
    ),
    ("warm start: u[i] = 0", "u[i] = low", "u[i] = 0"),
    (
        "warm start: never matches",
        "                p[j] = i\n                matched[i] = True\n                break",
        "                break",
    ),
    (
        "warm start: no empty-row check",
        "        if not allowed:\n            raise _Infeasible([i], [])\n",
        "",
    ),
    (
        "augment: stop condition inverted",
        "if p[j0] == -1:\n                break",
        "if p[j0] != -1:\n                break",
    ),
    ("path flip is a no-op", "p[j0] = p[j1]", "p[j0] = p[j0]"),
    ("flip loop ends early", "while j0 != m:", "while j0 != m and False:"),
    (
        "infeasible never raised",
        "if j1 < 0:\n                raise _Infeasible(",
        "if j1 < -9:\n                raise _Infeasible(",
    ),
    (
        "witness rows wrong",
        "sorted(p[j] for j in range(m + 1) if used[j]),",
        "sorted(p[j] for j in range(m + 1) if not used[j]),",
    ),
    ("wide/tall branch  <= -> <", "if n_rows <= n_cols:", "if n_rows < n_cols:"),
    ("result not sorted", "return sorted(pairs)", "return pairs"),
    (
        "transpose coords not swapped back",
        "return sorted((i, j) for j, i in pairs)",
        "return sorted((j, i) for j, i in pairs)",
    ),
    (
        "transpose edges use wrong cost",
        "by_column[j].append((i, cost))",
        "by_column[j].append((i, 0))",
    ),
    (
        "+inf not forbidden",
        "if is_pos_inf:\n                continue",
        "if False:\n                continue",
    ),
    ("-inf accepted", "if is_neg_inf:", "if False:"),
    (
        "NaN accepted",
        "if is_nan:\n                raise ValueError",
        "if False:\n                raise ValueError",
    ),
    ("ragged accepted", "if len(row) != width:", "if False:"),
    ("type check removed", "if not isinstance(value, _NUMBER_TYPES):", "if False:"),
    ("explain: single-row branch off", "if len(lefts) == 1 and not rights:", "if False:"),
    (
        "explain: swapped flag ignored",
        "rows, cols = (rights, lefts) if swapped else (lefts, rights)",
        "rows, cols = (lefts, rights)",
    ),
    (
        "make_cost_matrix: wrong sign",
        "inversion_function = lambda x: maximum - x",
        "inversion_function = lambda x: x - maximum",
    ),
    (
        "make_cost_matrix: inf not passed through",
        "(isinstance(value, float) and value == _INF)",
        "False",
    ),
    (
        "make_cost_matrix: DISALLOWED inverted",
        "value if forbidden(value) else inversion_function(value)",
        "inversion_function(value) if forbidden(value) else value",
    ),
    (
        "DISALLOWED not a singleton",
        "if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
        "return super().__new__(cls)",
    ),
    (
        "pad_matrix pads wrong amount",
        "row + [pad_value] * (total_rows - len(row))",
        "row + [pad_value] * (total_rows - len(row) - 1)",
    ),
    ("as_lists: no copy of rows", "rows.append(list(row))", "rows.append(row)"),
    ("print_matrix: width ignored", "fmt = '%%%d' % width", "fmt = '%%%d' % 0"),
    # ---- 2.0 gating / API / analysis / extras -------------------------------
    ("gate filter <= -> <", "if c <= gate", "if c < gate"),
    ("gate dummy cost off by one", "(m + k, gate)", "(m + k, gate + 1)"),
    ("gating keeps dummy pairs", "pairs = [(r, c) for r, c in pairs if c < m]", "pairs = pairs"),
    ("duals exported swapped", "duals.extend([u, v[:m]])", "duals.extend([v[:m], u])"),
    ("maximize: negation dropped", "new.append(-value)", "new.append(value)"),
    ("gate sign ignored", "gate = -threshold if maximize else threshold", "gate = threshold"),
    ("inf gate not disabled", "    if gate == _INF:\n        return None\n", ""),
    (
        "unmatched rows inverted",
        "unmatched_rows=tuple(i for i in range(n_rows) if i not in matched_rows)",
        "unmatched_rows=tuple(i for i in range(n_rows) if i in matched_rows)",
    ),
    (
        "labels swapped",
        "return tuple(index.tolist()), tuple(columns.tolist())",
        "return tuple(columns.tolist()), tuple(index.tolist())",
    ),
    ("bottleneck cap <= -> <", "if c <= limit]", "if c < limit]"),
    (
        "k_best: forced pairs dropped",
        "child_force = force + tuple(pairs[:t])",
        "child_force = force",
    ),
    (
        "tolerance sign flipped",
        "return solve(without).total - chosen.total",
        "return chosen.total - solve(without).total",
    ),
    (
        "shadow total ignores column prices",
        "Prices(tuple(rows_p), tuple(cols_p), sum(rows_p) + sum(cols_p))",
        "Prices(tuple(rows_p), tuple(cols_p), sum(rows_p))",
    ),
    (
        "stable: receiver preference inverted",
        "if rank[r][p] < rank[r][current]:",
        "if rank[r][p] > rank[r][current]:",
    ),
    (
        "transport: wrong demand units",
        "unit_cols = [j for j, d in enumerate(demand) for _ in range(d)]",
        "unit_cols = [j for j, d in enumerate(demand) for _ in range(d + 1)]",
    ),
    ("sinkhorn: tolerance ignored", "if err < tol:", "if err < 1.0:"),
]


def run(repo, target, only):
    rel = os.path.relpath(target, repo)
    src = open(target, encoding="utf-8").read()
    survivors, killed, skipped = [], 0, []
    todo = [m for m in MUTANTS if not only or m[0] in only]
    started = time.time()
    for label, old, new in todo:
        if src.count(old) != 1:
            skipped.append((label, src.count(old)))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "repo")
            shutil.copytree(
                repo,
                work,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".hypothesis",
                    "*.egg-info",
                    "build",
                    "dist",
                ),
            )
            path = os.path.join(work, rel)
            with open(path, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new))
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-x",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--timeout=40",
                    "-W",
                    "ignore",
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=600,
            )
        ok = proc.returncode != 0
        killed += ok
        print(("KILLED   " if ok else "SURVIVED ") + label, flush=True)
        if not ok:
            survivors.append(label)
    print(
        "\nmutants run: %d   killed: %d   survived: %d   skipped(not found once): %d   (%.0fs)"
        % (killed + len(survivors), killed, len(survivors), len(skipped), time.time() - started)
    )
    for label, n in skipped:
        print("  skipped:", label, "(found %d times)" % n)
    for label in survivors:
        print("  SURVIVOR:", label)
    return 1 if survivors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    sys.exit(run(a.repo, os.path.abspath(a.target), a.only))
