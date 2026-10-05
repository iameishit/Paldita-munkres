# Compatibility

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14 (CPython), tested in CI on Linux, plus Windows and macOS on 3.12.
- **Optional libraries**: numpy (>=1.26) and pandas (>=2.1) are recognised by duck typing (`tolist`, `to_numpy`,
  `index`, `columns`); nothing is imported unless you pass such an object.
- **SciPy**: `munkres.linear_sum_assignment(cost_matrix, maximize=False)` has the same arguments and result shape,
  raises `ValueError` for infeasible or invalid matrices, and treats `+inf` as forbidden. The total cost always
  matches; when several assignments tie, the pairs chosen may differ.
- **munkres 1.x**: `Munkres().compute()`, `pad_matrix()`, `make_cost_matrix()`, `print_matrix()`, `DISALLOWED`,
  `UnsolvableMatrix` behave as before for valid input. See [migration](migration.md) for the differences.
- **Typing**: ships `py.typed`; checked with `mypy --strict`.
