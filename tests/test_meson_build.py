"""The Meson build must stay in lock-step with the setuptools metadata (no Meson install needed)."""

import os
import re

import pytest

import munkres

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.packaging


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_meson_version_matches_the_package_version():
    assert re.search(r"version:\s*'%s'" % re.escape(munkres.__version__), _read("meson.build"))


def test_meson_lists_every_python_module_and_the_native_sources():
    text = _read("meson.build")
    listed = set(re.findall(r"'src/munkres/([\w]+\.py)'", text))
    on_disk = {n for n in os.listdir(os.path.join(ROOT, "src", "munkres")) if n.endswith(".py")}
    assert listed == on_disk, (
        f"meson.build out of date: missing {sorted(on_disk - listed)}, stale {sorted(listed - on_disk)}"
    )
    assert "src/munkres/_native/lapmodule.c" in text and "py.typed" in text
    assert os.path.exists(os.path.join(ROOT, "meson.options"))


def test_meson_pyproject_has_the_same_project_metadata_and_the_mesonpy_backend():
    setuptools_cfg = tomllib.loads(_read("pyproject.toml"))
    meson_cfg = tomllib.loads(_read("pyproject.meson.toml"))
    assert meson_cfg["project"] == setuptools_cfg["project"]
    assert meson_cfg["build-system"]["build-backend"] == "mesonpy"
    assert "setuptools" not in meson_cfg.get("tool", {})
