# Release checklist (stable releases)

- [ ] `tools/audit.sh` passes on every supported Python (the `Release` workflow runs the same checks)
- [ ] `python tools/verify_package.py` passes (build, metadata, clean-install smoke test)
- [ ] `python tools/verify_release.py --tag vX.Y.Z` passes (version, changelog, release notes, links)
- [ ] `src/munkres/__init__.py` holds the version; `CITATION.cff`, `meson.build` and the README badge match it
- [ ] `CHANGELOG.md`, `V3_RELEASE.txt` (or the matching release notes file) and `release/RELEASE_NOTES.md` describe the version
- [ ] `docs/BENCHMARKS.md` and `assets/benchmark.png` regenerated if performance claims changed
- [ ] The tag is a plain `vX.Y.Z` (no `rc`, `a`, `b` or `dev`); the `Release` workflow publishes it to PyPI
- [ ] A TestPyPI rehearsal was run from the manual `Release` workflow (workflow_dispatch) before tagging
- [ ] The GitHub release was created with the distributions and the source archive attached
