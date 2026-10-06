#!/usr/bin/env python
"""
Check that the repository is consistent and release-ready: required files, matching versions,
working links and images, valid configuration files, and public files free of internal notes.

    python tools/verify_release.py          # exit status 0 means no problems
    python tools/verify_release.py --tag v2.0.0      # also check a stable tag against the version
"""

import argparse
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import munkres  # noqa: E402

REQUIRED = [
    "README.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "LICENSE.md",
    "NOTICE",
    "V2_RELEASE.txt",
    "LICENSES/THIRD_PARTY_NOTICES.md",
    "AUTHORS.md",
    "MAINTAINERS.md",
    "CITATION.cff",
    "CODEOWNERS",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "SECURITY.md",
    "SUPPORT.md",
    "PUBLISHING.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "CODEX.md",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".python-version",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "justfile",
    "src/munkres/__init__.py",
    "src/munkres/__main__.py",
    "src/munkres/_core.py",
    "src/munkres/_api.py",
    "src/munkres/_analysis.py",
    "src/munkres/_extras.py",
    "src/munkres/_trace.py",
    "src/munkres/cli.py",
    "src/munkres/py.typed",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/helpers.py",
    "tests/test_munkres.py",
    "tests/test_bug_regressions.py",
    "tests/test_properties.py",
    "tests/test_fuzz_oracle.py",
    "tests/test_api.py",
    "tests/test_algorithms.py",
    "tests/test_cli.py",
    "tests/test_analysis.py",
    "tests/test_extras.py",
    "tests/test_trace.py",
    "tests/test_coverage_gaps.py",
    "tests/test_repo_hygiene.py",
    "tests/unit",
    "tests/integration",
    "tests/compatibility",
    "tests/numerical",
    "tests/constraints",
    "tests/concurrency",
    "tests/performance",
    "tests/security",
    "tests/fixtures",
    "tests/reference/munkres_reference.py",
    "tools/audit.sh",
    "tools/mutation_check.py",
    "tools/benchmark.py",
    "tools/verify_package.py",
    "tools/verify_release.py",
    "tools/test.py",
    "tools/lint.py",
    "tools/typecheck.py",
    "benchmarks/benchmark_dense.py",
    "benchmarks/benchmark_rectangular.py",
    "benchmarks/benchmark_constraints.py",
    "benchmarks/benchmark_memory.py",
    "benchmarks/benchmark_scaling.py",
    "benchmarks/benchmark_comparison.py",
    "examples/basic.py",
    "examples/rectangular.py",
    "examples/maximize.py",
    "examples/disallowed.py",
    "examples/numpy.py",
    "examples/analysis.py",
    "examples/trace.py",
    "examples/large_matrix.py",
    "docs/index.md",
    "docs/getting-started.md",
    "docs/installation.md",
    "docs/api.md",
    "docs/algorithm.md",
    "docs/architecture.md",
    "docs/numerical-contract.md",
    "docs/constraints.md",
    "docs/analysis.md",
    "docs/extras.md",
    "docs/tracing.md",
    "docs/performance.md",
    "docs/benchmarking.md",
    "docs/compatibility.md",
    "docs/migration.md",
    "docs/troubleshooting.md",
    "docs/development.md",
    "docs/release-process.md",
    "docs/security.md",
    "docs/faq.md",
    "docs/BENCHMARKS.md",
    "security/SECURITY_MODEL.md",
    "security/THREAT_MODEL.md",
    "security/SECURITY_TESTING.md",
    "release/RELEASE_CHECKLIST.md",
    "release/VERSIONING.md",
    "release/RELEASE_NOTES.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/documentation.yml",
    ".github/ISSUE_TEMPLATE/security.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/security.yml",
    ".github/workflows/dependency-review.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/benchmark.yml",
    ".github/workflows/nightly.yml",
    ".devcontainer/devcontainer.json",
    ".devcontainer/Dockerfile",
    ".well-known/security.txt",
    ".dockerignore",
    "assets/banner.jpg",
    "assets/logo.png",
    "assets/benchmark.png",
]

# Public files must describe the software for users: no internal notes or work-in-progress commentary.
FORBIDDEN = re.compile(
    r"\b(known (bugs?|issues?|limitations?)|internal plan|work[- ]in[- ]progress|TODO|FIXME|XXX|HACK)\b",
    re.I,
)
RAW = "https://raw.githubusercontent.com/iameishit/Paldita-munkres/main/"

problems: list[str] = []


def fail(message: str) -> None:
    problems.append(message)


def public_text_files():
    for pattern in (
        "*.md",
        "*.txt",
        "docs/*.md",
        "security/*.md",
        "release/*.md",
        ".github/**/*.md",
        ".github/**/*.yml",
        "src/munkres/*.py",
    ):
        yield from ROOT.glob(pattern)


def check_required():
    for name in REQUIRED:
        if not (ROOT / name).exists():
            fail(f"missing: {name}")


def check_versions(tag):
    version = munkres.__version__
    cff = (ROOT / "CITATION.cff").read_text()
    if f"version: {version}" not in cff:
        fail("CITATION.cff version differs from munkres.__version__")
    if f"Version {version}" not in (ROOT / "CHANGELOG.md").read_text():
        fail("CHANGELOG.md has no entry for the current version")
    if version not in (ROOT / "V2_RELEASE.txt").read_text():
        fail("V2_RELEASE.txt does not mention the current version")
    if version not in (ROOT / "release/RELEASE_NOTES.md").read_text():
        fail("release/RELEASE_NOTES.md does not mention the current version")
    if tag:
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            fail(f"tag {tag} is not a stable vX.Y.Z release tag")
        elif tag != f"v{version}":
            fail(f"tag {tag} does not match version v{version}")


def check_links():
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    for path in (
        list(ROOT.glob("*.md"))
        + list(ROOT.glob("docs/*.md"))
        + list(ROOT.glob("security/*.md"))
        + list(ROOT.glob("release/*.md"))
    ):
        for target in link.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                if target.startswith(RAW):
                    local = ROOT / target[len(RAW) :]
                    if not local.exists():
                        fail(f"{path.relative_to(ROOT)}: image not found in repo: {target}")
                continue
            if not (path.parent / target.split("#")[0]).exists():
                fail(f"{path.relative_to(ROOT)}: broken link {target}")
    readme = (ROOT / "README.md").read_text()
    for needed in ("assets/banner.jpg", "assets/logo.png", "assets/benchmark.png"):
        if RAW + needed not in readme:
            fail(f"README.md does not show {needed}")


def check_wording():
    for path in public_text_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                fail(
                    f"{path.relative_to(ROOT)}:{number}: internal-style wording: {line.strip()[:70]}"
                )


def check_config_files():
    try:
        import yaml
    except ImportError:
        print("(pyyaml not installed: skipping YAML validation)")
        yaml = None
    if yaml:
        for path in list(ROOT.glob(".github/**/*.yml")) + [
            ROOT / "CITATION.cff",
            ROOT / ".pre-commit-config.yaml",
        ]:
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                fail(f"{path.relative_to(ROOT)}: invalid YAML ({error})")
        cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        for key in (
            "cff-version",
            "message",
            "title",
            "authors",
            "version",
            "license",
            "repository-code",
        ):
            if key not in cff:
                fail(f"CITATION.cff lacks {key}")
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            if "jobs" not in doc or (True not in doc and "on" not in doc):
                fail(f"{path.name}: not a workflow (needs `on` and `jobs`)")
    try:
        json.loads((ROOT / ".devcontainer/devcontainer.json").read_text(encoding="utf-8"))
    except ValueError as error:
        fail(f".devcontainer/devcontainer.json: invalid JSON ({error})")
    text = (ROOT / ".well-known/security.txt").read_text()
    expires = re.search(r"^Expires:\s*(\S+)", text, re.M)
    if not (expires and re.search(r"^Contact:", text, re.M)):
        fail(".well-known/security.txt needs Contact and Expires (RFC 9116)")
    elif datetime.datetime.fromisoformat(
        expires.group(1).replace("Z", "+00:00")
    ) < datetime.datetime.now(datetime.timezone.utc):
        fail(".well-known/security.txt has expired")


def check_api_docs():
    api = (ROOT / "docs/api.md").read_text()
    for name in munkres.__all__:
        if f"`{name}" not in api:
            fail(f"docs/api.md does not document {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()
    check_required()
    check_versions(args.tag)
    check_links()
    check_wording()
    check_config_files()
    check_api_docs()
    for message in problems:
        print("PROBLEM:", message)
    print("RELEASE CHECKS PASSED" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
