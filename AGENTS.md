# Guidance for coding agents

munkres is a dependency-free, fully typed, pure-Python library for the assignment problem.
Source lives in `src/munkres/`; tests in `tests/`.

## Commands

```bash
pip install -e . --group dev        # or: uv sync
python -m pytest                    # full test-suite (hangs fail fast)
tools/audit.sh --quick              # lint, format, types, security, coverage, build, smoke test
tools/audit.sh                      # ... plus every supported Python (needs uv)
```

## Rules

1. Write the failing test first. New algorithms are checked against an **independent**
   oracle (`tests/helpers.py`, `tests/reference/munkres_reference.py`, SciPy), never against
   themselves.
2. Keep 100 % line and branch coverage and `mypy --strict` clean.
3. No runtime dependencies. numpy and pandas stay optional and are only duck-typed.
4. Never mutate the caller's input; instances hold no state (thread-safe by design).
5. Python 3.10+ syntax only. Keep public names in `munkres/__init__.py` and in `__all__`.
6. Public files (README, docs, changelog, issue templates) describe what the software does for users;
   keep personal notes and scratch commentary out of them.
7. Add a `CHANGELOG.md` line for every user-visible change.

## Layout

`_core.py` solver and validation, `_api.py` `solve()`/`Assignment`, `_analysis.py` prices,
what-ifs, k-best, bottleneck, `_extras.py` stable matching, transport, Sinkhorn, `_trace.py`
step traces, `cli.py` command line.
