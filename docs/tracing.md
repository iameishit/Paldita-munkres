# Tracing

```python
from munkres import solve
trace = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True).trace
print(trace.to_text())
open("trace.html", "w").write(trace.to_html())
```

`trace.events` is the raw list of steps (`row_reduce`, `star`, `augment_start`, `adjust`, `augmented`, `done`);
`trace.steps()` gives one plain-language sentence per step; `to_html()` produces a static, script-free page.
For a matrix with more rows than columns the solver works on the transpose; `trace.transposed` is then true and the
text already uses the right words.

From the command line: `munkres costs.csv --trace` or `--trace-html trace.html`.
