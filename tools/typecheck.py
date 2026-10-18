#!/usr/bin/env python
"""Run mypy --strict on the package (cross-platform wrapper)."""

import subprocess
import sys

raise SystemExit(
    subprocess.run([sys.executable, "-m", "mypy", *sys.argv[1:]], check=False).returncode
)
