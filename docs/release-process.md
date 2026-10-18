# Release process

1. `tools/audit.sh` passes, `python tools/verify_package.py` passes, and
   `python tools/verify_release.py --tag vX.Y.Z` passes.
2. The version is updated in `src/munkres/__init__.py`; `CITATION.cff`, `meson.build`, the README badge,
   `CHANGELOG.md` and the release notes match it.
3. A TestPyPI rehearsal runs from the manual `Release` workflow.
4. The maintainer pushes the plain `vX.Y.Z` tag. The `Release` workflow checks the tag, builds the wheels and
   source distribution, waits for approval in the `pypi` environment, publishes to PyPI and creates the GitHub
   release.

Details: [PUBLISHING.md](../PUBLISHING.md), [release/RELEASE_CHECKLIST.md](../release/RELEASE_CHECKLIST.md),
[release/VERSIONING.md](../release/VERSIONING.md).
