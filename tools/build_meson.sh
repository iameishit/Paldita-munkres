#!/bin/sh
# Build a wheel with Meson (meson-python) without touching the setuptools configuration.
#   sh tools/build_meson.sh [outdir]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist-meson}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -r "$ROOT/." "$TMP/src-tree"
cd "$TMP/src-tree"
cp pyproject.meson.toml pyproject.toml
python3 -m pip wheel . --no-build-isolation --no-deps -w "$OUT" || python3 -m pip wheel . --no-deps -w "$OUT"
echo "wheel(s) in $OUT"
