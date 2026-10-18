# Threat model

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Resource exhaustion | very large matrix, or constraints that cannot be satisfied | work bounded by O(n² · m); infeasible and non-finite input rejected immediately; callers bound matrix size for untrusted input |
| Malicious serialized data | unpickling a crafted object | `DISALLOWED` pickles as a global reference; the library never unpickles anything itself |
| Code execution through input | crafted cell values | cells are only compared and added; no `eval`/`exec`; non-numeric values raise `TypeError` |
| Information disclosure | error messages | messages contain indexes and the offending value, nothing else |
| Hostile file to the CLI | path or format abuse | the CLI opens only the path it is given, parses numbers only, and reports errors with exit codes |
| Tampered release | stolen credentials, malicious dependency | trusted publishing, 2FA, no runtime dependencies, dependency review |
| Typosquatting | look-alike package names | documented canonical project URL and install command |
