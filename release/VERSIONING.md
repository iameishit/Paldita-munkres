# Versioning

munkres follows [Semantic Versioning](https://semver.org/) with plain `MAJOR.MINOR.PATCH` version numbers.

- **Major**: incompatible changes to the public API (everything in `munkres.__all__` and the command line).
- **Minor**: new backwards-compatible features.
- **Patch**: backwards-compatible fixes.

Stable releases are tagged `vMAJOR.MINOR.PATCH` (for example `v3.0.0`) and published to PyPI. Pre-release
spellings are not used for publication; rehearsals go to TestPyPI through the manual `Release` workflow.
The single source of the version is `munkres.__version__`.
