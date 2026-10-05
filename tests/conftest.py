"""Shared fixtures and automatic markers."""

import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).parent

# Tests start the command line, the examples and the benchmarks as child processes. Make the source
# tree importable for them too, so the suite also passes in an environment where the package is not
# installed.
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(HERE.parent / "src"), os.environ.get("PYTHONPATH")) if part
)


@pytest.fixture
def fixtures_dir():
    return HERE / "fixtures"


@pytest.fixture
def fixture_file(fixtures_dir):
    """Return the path (as a string) of a file in tests/fixtures."""

    def get(name):
        return str(fixtures_dir / name)

    return get


def pytest_collection_modifyitems(items):
    """Mark tests by the directory they live in, so `-m performance` etc. work."""
    for item in items:
        for area in ("performance", "security"):
            if f"/{area}/" in str(item.fspath).replace("\\", "/"):
                item.add_marker(getattr(pytest.mark, area))
