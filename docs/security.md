# Security

munkres performs no network access, runs no external programs, evaluates no code and keeps no global state. Its
only I/O is `print_matrix` writing to stdout and the optional command line tool reading the file you name. Work
per call is bounded by O(n² · m); as with any numeric library, apply your own size limits to untrusted input.

`DISALLOWED` pickles as a plain reference to a module global, so unpickling cannot run attacker-chosen code through
this type. Releases are published with PyPI trusted publishing.

Report vulnerabilities privately: see [SECURITY.md](../SECURITY.md). Background: [security/SECURITY_MODEL.md](../security/SECURITY_MODEL.md),
[security/THREAT_MODEL.md](../security/THREAT_MODEL.md), [security/SECURITY_TESTING.md](../security/SECURITY_TESTING.md).


## v3: native code and hostile inputs

- **Native boundary.** The only compiled code is `src/munkres/_native/` (a single C file wrapped as a CPython extension). It receives buffers only through the buffer protocol; the wrapper checks every size before the kernel runs, rejects non-contiguous, read-only-output and wrong-size buffers, and never reads or writes outside the buffers it was given. `tests/hard/test_kernel_hard.py` throws thousands of random argument combinations at it.
- **Numeric safety.** The kernel rejects NaN and -infinity, and declines (the caller then solves exactly) when `max|c| * (n + m + 4)` could overflow, so there is no silent integer or float overflow.
- **No unsafe deserialisation.** Certificates and specs are JSON only. `Certificate.from_json` and `solve_spec` validate every field and raise `ValueError` for anything malformed (fuzzed in `tests/hard/test_system_hard.py`); nothing is evaluated or unpickled.
- **Resource limits.** `SolverLimits` / `solve_limited` / the `limits` key of `solve_spec` refuse oversized problems from their shape alone, before any allocation, with a structured `ProblemTooLargeError` (`.code`, `.details`).
- **Threads.** The kernel releases the GIL and holds no global mutable state; concurrent solves of shared read-only inputs are tested. `AssignmentState` and `AssignmentSolver` instances are not thread-safe: use one per thread.
- **Not covered.** No third-party security audit; no fuzzing with sanitizers (ASan/UBSan) was possible in the development environment. Run the extension under sanitizers before relying on it with untrusted data at scale.
