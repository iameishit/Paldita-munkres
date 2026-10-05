# Security

munkres performs no network access, runs no external programs, evaluates no code and keeps no global state. Its
only I/O is `print_matrix` writing to stdout and the optional command line tool reading the file you name. Work
per call is bounded by O(n² · m); as with any numeric library, apply your own size limits to untrusted input.

`DISALLOWED` pickles as a plain reference to a module global, so unpickling cannot run attacker-chosen code through
this type. Releases are published with PyPI trusted publishing.

Report vulnerabilities privately: see [SECURITY.md](../SECURITY.md). Background: [security/SECURITY_MODEL.md](../security/SECURITY_MODEL.md),
[security/THREAT_MODEL.md](../security/THREAT_MODEL.md), [security/SECURITY_TESTING.md](../security/SECURITY_TESTING.md).
