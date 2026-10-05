# Publishing to PyPI

Releases are built and uploaded by GitHub Actions (`.github/workflows/release.yml`)
using **PyPI trusted publishing**, so no API token is stored anywhere.

## One-time setup (needs PyPI owner access to the `munkres` project)

1. PyPI -> project `munkres` -> *Publishing* -> add a trusted publisher:
   owner `iameishit`, repository `Paldita-munkres`, workflow `release.yml`,
   environment `pypi`. Do the same on TestPyPI with environment `testpypi`.
2. GitHub -> repository *Settings -> Environments*: create `pypi` and
   `testpypi` (add a required reviewer to `pypi` for a manual approval gate).
3. Turn on 2FA for every PyPI owner.

## Cutting a release

```bash
tools/audit.sh                      # everything must PASS
# bump __version__ in src/munkres/__init__.py, update CHANGELOG.md, commit
git tag v2.0.0rc1 && git push --tags      # tags containing "rc"/"dev" go to TestPyPI
```

Pre-release tags (`rc`, `a`, `b`, `dev`) publish to TestPyPI; plain `vX.Y.Z`
tags publish to PyPI. Verify a TestPyPI release with
`pip install -i https://test.pypi.org/simple/ munkres==2.0.0rc1` first.

`pyproject.toml` (the import name stays `munkres`).

## Pointing 1.x users at 2.x

Before the first 2.x release, a final `1.1.5` whose only change is a notice helps people
who cannot upgrade immediately. Suggested wording for its README / PyPI description:

> munkres 1.x is end-of-life. 2.0 fixes hangs on impossible or NaN input, adds typing,
> numpy/pandas input and many new features, and requires Python 3.10+. Pin `munkres<2`
> to stay on 1.x; see CHANGELOG.md for the (small) list of breaking changes.
