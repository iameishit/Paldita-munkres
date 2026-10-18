# FAQ

**Is it thread-safe?** Yes. Objects hold no state; share one `Munkres` instance freely.

**Does it modify my matrix?** Never, including numpy arrays.

**What if there are more rows than columns?** `min(rows, columns)` pairs are returned; the rest are reported as unmatched.

**How do I maximise?** `solve(profit, maximize=True)`, or `make_cost_matrix(profit)` with `Munkres().compute()`.

**How do I forbid a pairing?** `DISALLOWED` or `float("inf")`.

**Why pure Python?** Zero dependencies, exact `Decimal`/`Fraction` support and clear errors. A compiled solver such as
SciPy is faster on very large dense problems; `linear_sum_assignment` has the same signature.

**Which Python versions?** 3.10 to 3.14.

**License?** Apache 2.0, originally by Brian M. Clapper.

**How do I cite it?** See `CITATION.cff`.
