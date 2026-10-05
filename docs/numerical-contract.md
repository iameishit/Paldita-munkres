# Numerical contract

| Input | Behaviour |
|-------|-----------|
| `int`, `float`, `Fraction`, `Decimal`, `bool` | accepted; arithmetic is exact for exact types |
| integers of any size | accepted |
| numpy / pandas values | converted to Python values first |
| `DISALLOWED`, `+inf` | pairing forbidden |
| `-inf`, `NaN` | `ValueError` |
| `+inf` profit with `maximize=True` | `ValueError`; use `-inf` or `DISALLOWED` to forbid |
| anything else | `TypeError` |
| ragged rows | `ValueError` |
| no rows or no columns | empty result |

- Totals are computed from your original values, in your original types.
- When several assignments are equally optimal, any one may be returned; the total is the same.
- Maximising negates values instead of subtracting from a maximum, so it is exact for every numeric type.
- Float inputs follow ordinary floating-point rounding; use `Decimal` or `Fraction` for exact results.
