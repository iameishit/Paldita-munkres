# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest pytest-timeout pytest-cov hypothesis numpy scipy pandas "tomli; python_version<'3.11'" \
    ruff mypy bandit vulture codespell pdoc build twine check-wheel-contents
python -m pytest           # tests (hangs fail fast thanks to a per-test timeout)
tools/audit.sh --quick     # lint, types, security, dead code, coverage, build, smoke test
tools/audit.sh             # ... plus the test-suite on every supported Python (needs uv)
```

Rules of the road:

1. **A bug fix starts with a failing test.** Add it to
   `tests/test_bug_regressions.py` and watch it fail first.
2. New algorithms must be checked against an *independent* oracle
   (`tests/helpers.py`: exhaustive search and SciPy), never against themselves.
3. Coverage must stay at 100 % (line and branch); `mypy --strict` must pass.
4. Add a line to `CHANGELOG.md` for every user-visible change.
