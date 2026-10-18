# Performance

munkres is pure Python and runs in O(n² · m) time. A dense 200x200 problem takes about a tenth of a second; see
[BENCHMARKS.md](BENCHMARKS.md) for measured numbers (reproduce with `python tools/benchmark.py`).

Tips:

- Rectangular problems cost O(n² · m) with n the smaller side: tall or wide matrices are cheap.
- Forbidden cells are not stored, so sparse problems do proportionally less work.
- Matrices with many ties are solved quickly thanks to the warm start.
- For very large dense problems where raw speed matters most, a compiled solver such as
  `scipy.optimize.linear_sum_assignment` is faster; `munkres.linear_sum_assignment` has the same call signature
  so switching is a one-line change either way.
