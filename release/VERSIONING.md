# Versioning

munkres follows [Semantic Versioning](https://semver.org/) with [PEP 440](https://peps.python.org/pep-0440/)
spellings (`2.0.0`).

- **Major**: incompatible changes to the public API (everything in `munkres.__all__` and the command line).
- **Minor**: new backwards-compatible features.
- **Patch**: backwards-compatible fixes.
- **Pre-releases** (`aN`, `bN`, `rcN`) are not accepted by the stable PyPI release workflow; they require
  a separately configured publishing workflow.

Deprecated features are announced in the changelog and kept for at least one minor release before removal.
The single source of the version is `munkres.__version__`.
