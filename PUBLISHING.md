# Publishing to PyPI

Releases are built and uploaded by the `Release` workflow (`.github/workflows/release.yml`) using **PyPI
trusted publishing**, so no API token is stored anywhere.

## One-time setup (maintainer with owner access to the `munkres` project)

1. PyPI -> project `munkres` -> *Publishing* -> add a trusted publisher: owner `iameishit`, repository
   `Paldita-munkres`, workflow `release.yml`, environment `pypi`. Add a matching publisher on TestPyPI with
   environment `testpypi`.
2. GitHub -> repository *Settings -> Environments*: create `pypi` and `testpypi`. Add a required reviewer to
   `pypi` so every real release is approved by a maintainer.
3. Turn on two-factor authentication for every PyPI owner.

## Releasing 3.0.0

```bash
python tools/verify_release.py --tag v3.0.0   # must pass
git tag v3.0.0
git push origin v3.0.0
```

The workflow builds the source distribution and the platform wheels, checks them, publishes to PyPI (after
approval), and attaches the distributions and a source archive to the GitHub release.

## Rehearsing on TestPyPI

Run the `Release` workflow manually (Actions -> Release -> Run workflow). It builds and checks everything and
uploads to TestPyPI only. Verify with:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ munkres==3.0.0
```

Tags that are not plain `vX.Y.Z` (for example `v3.0.0rc1`) are refused by the workflow.
