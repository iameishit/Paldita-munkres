# Migrating from 1.x

Most code needs no change. Differences:

| 1.x | 2.x |
|-----|-----|
| Python 2/3 wheel tag, Python 3.5+ | Python 3.10+ |
| ragged matrices accepted (and mis-solved) | `ValueError` |
| empty matrix: `IndexError` or `[]` | `[]` |
| `NaN` / `-inf` could hang | `ValueError` |
| `+inf` cost | forbidden, like `DISALLOWED` |
| impossible `DISALLOWED` pattern could hang | `UnsolvableMatrix` (a `ValueError`) with `.rows` / `.cols` |
| `UnsolvableMatrix` derived from `Exception` | derives from `ValueError`; `except UnsolvableMatrix` still works |
| numpy input modified in place | never modified |
| `setup.py` | `pyproject.toml` |

New code can use `solve()` for a richer result; `Munkres().compute()` remains fully supported.

```python
# 1.x style, still fine
from munkres import Munkres
pairs = Munkres().compute(matrix)

# 2.x style
from munkres import solve
result = solve(matrix)
```
