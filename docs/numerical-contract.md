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


## v3 additions

- **Native kernel domain.** The C kernel accepts float64 and int64 only, rejects NaN and -inf, and declines (returns a code; Python re-solves exactly) when `max|c| * (n + m + 4)` could overflow (about 1.7e308 for float64, 4e18 for int64). For int64 results are exact. For float64 the kernel uses ordinary IEEE arithmetic: it finds an optimum of the float problem with float rounding in the dual updates, the same as the pure-Python float path.
- **Float overflow.** Where float arithmetic could overflow, the pure-Python solver switches to exact `Fraction` arithmetic (v2.0.0 raised a false `UnsolvableMatrix` there).
- **Exact tools.** `certify`, `topology`, `enumerate_optima`, `canonical_optimum`, `AssignmentState`, `regret_of` treat float costs as the exact rationals they are; their guarantees hold for the rational problem.
- **Determinism.** The solver's own tie-breaking is deterministic but not specified; use `canonical_optimum` for a defined choice (lexicographically smallest optimum).
- **Certificates** prove optimality of the stated pairs for the exact problem identified by `problem_hash`; they say nothing about float rounding in a float64 solver run (the verifier re-derives its own exact duals).
