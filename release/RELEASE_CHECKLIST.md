# Release checklist

- [ ] `tools/audit.sh` passes (all Python versions)
- [ ] `python tools/verify_package.py` passes (build, metadata, clean-install smoke test)
- [ ] `python tools/verify_release.py` passes (version, changelog, notes, links)
- [ ] `src/munkres/__init__.py` version updated; `CITATION.cff` version matches
- [ ] `CHANGELOG.md`, `V2_RELEASE.txt` / `release/RELEASE_NOTES.md` updated
- [ ] `docs/BENCHMARKS.md` regenerated if performance changed (`python tools/benchmark.py`)
- [ ] Tag pushed (`vX.Y.Z`); `Release` workflow green; TestPyPI install verified for pre-releases
- [ ] GitHub release created from the notes
