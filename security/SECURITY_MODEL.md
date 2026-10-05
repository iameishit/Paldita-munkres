# Security model

## What the library does

Pure computation on a matrix the caller provides. It reads no files (except the file named to the optional command
line tool), opens no connections, starts no processes, evaluates no code, and stores no global state.

## Guarantees

- **Termination.** Every call performs a bounded amount of work: at most O(n² · m) steps. NaN, infinities and
  impossible constraint patterns are rejected or reported up front.
- **Isolation.** Inputs are copied before use and never modified; objects hold no state, so concurrent use is safe.
- **No code execution.** There is no use of `eval`, `exec`, `subprocess`, `os.system`, `ctypes` or `socket` in the
  package (enforced by a test), and unpickling `DISALLOWED` resolves to a module global only.
- **Zero runtime dependencies**, so there is no transitive supply-chain surface.

## Trust boundaries

Matrix contents and sizes are data from the caller. If they come from an untrusted source, the caller should bound the
matrix size, because running time grows with it.

## Supply chain

Releases are built in CI and published with PyPI trusted publishing (no long-lived token); maintainers use two-factor
authentication; CI runs CodeQL, dependency review, bandit, pip-audit and a secrets scan.
