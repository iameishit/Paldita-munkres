# Security testing

| Check | Where | What it covers |
|-------|-------|----------------|
| `tests/security/` | pytest | hostile inputs (huge ints, NaN/inf, deep nesting, junk types), no dangerous calls or network/process imports in `src/`, safe pickling |
| bandit | `tools/audit.sh`, `security.yml` | static analysis of the shipped package |
| secrets scan | `tools/audit.sh`, `security.yml` | detect-secrets over tracked files |
| pip-audit | `security.yml` | known vulnerabilities in the environment |
| CodeQL | `codeql.yml` | semantic analysis on every push and weekly |
| dependency review | `dependency-review.yml` | new dependencies in pull requests |
| fuzzing | `tests/test_fuzz_oracle.py`, `tests/test_properties.py` | randomised inputs against independent oracles |

Run locally: `python -m pytest tests/security` and `tools/audit.sh --quick`.
