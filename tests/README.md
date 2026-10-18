# Tests

```bash
python -m pytest                     # everything (about a minute)
python -m pytest tests/security      # one area
python -m pytest -m performance
python -m pytest --cov               # with the 100 % line and branch coverage gate
```

| Where | What |
|-------|------|
| `test_munkres.py` | the original examples, validated against independent oracles |
| `test_bug_regressions.py` | behaviours fixed in 2.x (hangs, ragged/empty input, numpy, threads, pickling ...) |
| `test_properties.py`, `test_fuzz_oracle.py` | randomised tests against exhaustive search and SciPy |
| `test_api.py`, `test_trace.py`, `test_analysis.py`, `test_extras.py`, `test_cli.py` | the high-level API, traces, analysis tools, related problems, the command line |
| `test_coverage_gaps.py`, `test_repo_hygiene.py` | mocks and edge cases; packaging, docs and structure checks |
| `unit/` | validation, the sentinel, the 1.x helper functions |
| `integration/` | features working together; the command line end to end |
| `compatibility/` | the 1.x API; SciPy parity |
| `numerical/` | exact arithmetic, huge values, invariances |
| `constraints/` | forbidden cells (Hall's condition) and gating |
| `concurrency/` | threads and processes |
| `performance/` | time and memory budgets |
| `security/` | hostile input, no dangerous calls or imports |
| `reference/` | an independent reference solver (subset DP) used as an oracle |
| `fixtures/` | sample matrix files |
| `helpers.py`, `conftest.py` | oracles, validators, hang guard; shared fixtures |
