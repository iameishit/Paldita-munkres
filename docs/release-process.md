# Release process

1. `tools/audit.sh`, `python tools/verify_package.py`, and `python tools/verify_release.py --tag v2.0.0` pass.
2. `src/munkres/__init__.py` is the single version source; `CITATION.cff`, `CHANGELOG.md`,
   `V2_RELEASE.txt`, and the release notes agree with it.
3. An authorized maintainer pushes the stable `v2.0.0` tag. The release workflow verifies that it matches the
   package version, builds and checks clean distributions, then publishes to real PyPI with Trusted Publishing.

Details: [PUBLISHING.md](../PUBLISHING.md), [release/RELEASE_CHECKLIST.md](../release/RELEASE_CHECKLIST.md),
[release/VERSIONING.md](../release/VERSIONING.md).
