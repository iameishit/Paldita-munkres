#!/usr/bin/env python
"""Run the test-suite (cross-platform wrapper): `python tools/test.py [pytest args]`."""

import subprocess
import sys

raise SystemExit(
    subprocess.run([sys.executable, "-m", "pytest", *sys.argv[1:]], check=False).returncode
)
