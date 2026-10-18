#!/usr/bin/env python
"""Lint and format-check everything (cross-platform wrapper): `python tools/lint.py [--fix]`."""

import subprocess
import sys

DIRS = ["src", "tests", "tools", "benchmarks", "examples"]
fix = "--fix" in sys.argv
status = 0
if fix:
    status |= subprocess.run(["ruff", "check", "--fix", *DIRS], check=False).returncode
    status |= subprocess.run(["ruff", "format", *DIRS], check=False).returncode
else:
    status |= subprocess.run(["ruff", "check", *DIRS], check=False).returncode
    status |= subprocess.run(["ruff", "format", "--check", *DIRS], check=False).returncode
raise SystemExit(status)
