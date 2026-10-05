# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** through GitHub's
"Report a vulnerability" button (Security tab of this repository) rather than
a public issue. You will get an acknowledgement within 7 days.

## Scope and design notes

- `munkres` has **no runtime dependencies**, performs no I/O (other than
  `print_matrix` writing to stdout and the optional CLI reading the file you
  name), opens no network connections, and uses no `eval`, `exec` or
  `subprocess`.
- Pickling is safe: `DISALLOWED` pickles as a plain reference to a module
  global, so unpickling cannot execute attacker-chosen code through this type.
- **Denial of service**: every public call terminates in a bounded number of
  steps (at most O(n^2 * m) work). Unsolvable, NaN and infinite inputs are
  rejected immediately; they cannot hang the solver. Very large matrices are
  still expensive in pure Python, so enforce size limits before calling the
  library with untrusted input.

## Supported versions

Only the latest 2.x release receives fixes. 1.x is end-of-life.
