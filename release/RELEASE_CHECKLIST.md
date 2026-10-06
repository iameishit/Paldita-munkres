# Release checklist

- [ ] `tools/audit.sh` passes (all Python versions)
- [ ] `python tools/verify_package.py` passes (build, metadata, clean-install smoke test)
- [ ] `python tools/verify_release.py --tag v2.0.0` passes (version, changelog, notes, links)
- [ ] `src/munkres/__init__.py` is the single version source; `CITATION.cff` matches `2.0.0`
- [ ] `CHANGELOG.md`, `V2_RELEASE.txt`, and `release/RELEASE_NOTES.md` describe stable `2.0.0`
- [ ] `docs/BENCHMARKS.md` regenerated if performance changed (`python tools/benchmark.py`)
- [ ] Stable `v2.0.0` tag pushed by an authorized maintainer; release workflow green on real PyPI
- [ ] GitHub release created from the notes
