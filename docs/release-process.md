# Release process

1. `tools/audit.sh` passes and `python tools/verify_release.py` reports no problems.
2. Version bumped in `src/munkres/__init__.py`; `CHANGELOG.md`, `V2_RELEASE.txt`/release notes updated.
3. Tag `vX.Y.Z` (pre-releases like `vX.Y.ZrcN` go to TestPyPI first). The `Release` workflow builds, checks and
   publishes with PyPI trusted publishing.

Details: [PUBLISHING.md](../PUBLISHING.md), [release/RELEASE_CHECKLIST.md](../release/RELEASE_CHECKLIST.md),
[release/VERSIONING.md](../release/VERSIONING.md).
