# Publishing to PyPI

Releases are built and uploaded by GitHub Actions (`.github/workflows/release.yml`)
using **PyPI trusted publishing**, so no API token is stored anywhere.

## One-time setup (needs PyPI owner access to the `munkres` project)

1. PyPI -> project `munkres` -> *Publishing* -> add a trusted publisher:
   owner `iameishit`, repository `Paldita-munkres`, workflow `release.yml`,
   environment `pypi`.
2. GitHub -> repository *Settings -> Environments*: create `pypi` (add a
   required reviewer for a manual approval gate).
3. Turn on 2FA for every PyPI owner.

## Stable release

```bash
tools/audit.sh
python tools/verify_package.py
python tools/verify_release.py --tag v2.0.0
```

The release workflow runs for `v*` tags, verifies that the tag is a stable
`vX.Y.Z` matching the package version, builds clean distributions, checks them
with Twine and publishes to the real PyPI using Trusted Publishing/OIDC and the
`pypi` environment. Pre-release tags are rejected by this workflow; publishing
pre-releases requires a separately configured workflow.

Project metadata is in `pyproject.toml`; the import name stays `munkres`.

## Supporting 1.x users

If a final 1.x maintenance release is needed, a notice can help users who cannot
upgrade immediately. Suggested wording for its README / PyPI description:

> munkres 1.x is end-of-life. 2.0 fixes hangs on impossible or NaN input, adds typing,
> numpy/pandas input and many new features, and requires Python 3.10+. Pin `munkres<2`
> to stay on 1.x; see CHANGELOG.md for the (small) list of breaking changes.
