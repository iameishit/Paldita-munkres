"""Packaging, metadata, README and public-API hygiene (audit items B15-B21)."""

import inspect
import os
import re

import pytest

import munkres

try:  # tomllib is stdlib from Python 3.11; 3.10 needs the `tomli` backport
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.packaging


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def pyproject():
    return tomllib.loads(read("pyproject.toml"))


def test_B15_all_exports_public_api():
    assert {
        "Munkres",
        "make_cost_matrix",
        "DISALLOWED",
        "UnsolvableMatrix",
        "print_matrix",
    } <= set(munkres.__all__)


def test_every_exported_class_and_function_is_documented():
    for name in munkres.__all__:
        obj = getattr(munkres, name)
        if isinstance(obj, type) or inspect.isfunction(obj):
            assert obj.__doc__, f"{name} has no docstring"


def test_B17_py_typed_marker_in_package():
    assert os.path.exists(os.path.join(ROOT, "src", "munkres", "py.typed"))
    assert pyproject()["tool"]["setuptools"]["package-data"]["munkres"] == ["py.typed"]


def test_B18_no_legacy_packaging_files():
    assert not os.path.exists(os.path.join(ROOT, "setup.py"))
    assert not os.path.exists(os.path.join(ROOT, "setup.cfg"))
    assert not os.path.exists(os.path.join(ROOT, ".travis.yml"))


def test_B18_python_requires_declared_and_py3_only():
    project = pyproject()["project"]
    assert project["requires-python"] == ">=3.10"
    assert "Programming Language :: Python :: 2" not in project["classifiers"]
    assert "Typing :: Typed" in project["classifiers"]
    assert project["dependencies"] == []  # zero runtime dependencies, by design


def test_classifier_versions_match_ci_matrix():
    ci = read(".github/workflows/ci.yml")
    classifiers = pyproject()["project"]["classifiers"]
    for version in re.findall(r'"(3\.\d+)"', ci.split("test:")[1].split("steps:")[0]):
        assert f"Programming Language :: Python :: {version}" in classifiers


def test_B19_copyright_years_consistent():
    lic = re.findall(r"2008-(\d{4})", read("LICENSE.md"))
    mod = re.findall(r"2008-(\d{4})", munkres.__copyright__)
    assert lic and mod and lic[0] == mod[0] == "2020"


def test_B19_notice_credits_original_author_and_new_maintainer():
    notice = read("NOTICE")
    assert "Brian M. Clapper" in notice and "2026" in notice and "Eishit Nigam" in notice


def test_license_files_declared_in_pyproject_exist():
    for name in pyproject()["project"]["license-files"]:
        assert os.path.exists(os.path.join(ROOT, name)), name


def test_B21_readme_complete():
    text = read("README.md")
    assert text.count("```") % 2 == 0, "unclosed code fence"
    assert "pip install" in text
    assert len(re.findall(r"^# ", text, flags=re.M)) == 1, "exactly one H1"
    assert "Current Version" not in text, "stale hard-coded version section"


def test_readme_python_examples_run_and_print_what_the_comments_claim(capsys):
    """Every ```python block must run, and each `print(...)  # expected` comment
    must match the real output (a trailing `...` means "starts with")."""
    text = read("README.md")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, "README has no python examples"
    for block in blocks:
        expected = [
            m.group(1).strip()
            for line in block.splitlines()
            if "print(" in line and (m := re.search(r"#\s*(.+)$", line))
        ]
        capsys.readouterr()
        exec(compile(block, "<README>", "exec"), {})  # noqa: S102
        printed = capsys.readouterr().out.splitlines()
        assert len(printed) >= len(expected)
        for want, got in zip(expected, printed, strict=False):
            if want.endswith("..."):
                assert got.startswith(want[:-3].rstrip()), (want, got)
            else:
                assert got == want, (want, got)


def test_version_is_single_sourced_and_in_changelog():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+|\.dev\d+)?", munkres.__version__)
    assert pyproject()["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "munkres.__version__"
    }
    assert f"Version {munkres.__version__}" in read("CHANGELOG.md")


def test_security_and_contributing_docs_present():
    for name in (
        "SECURITY.md",
        "CONTRIBUTING.md",
        "PUBLISHING.md",
        ".github/workflows/release.yml",
    ):
        assert os.path.exists(os.path.join(ROOT, name)), name


def test_release_verifier_reports_no_problems():
    """Required files, versions, links, images, config files and public wording (tools/verify_release.py)."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "verify_release.py")],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
