#!/usr/bin/env bash
# One command, every check. Exit status 0 only if everything passes.
#   tools/audit.sh --quick   skip the multi-Python test matrix (needs `uv`)
set -u
cd "$(dirname "$0")/.."
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
FAILED=()
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; CURRENT="$1"; }
check() { if "$@"; then echo "   PASS"; else echo "   FAIL: $CURRENT"; FAILED+=("$CURRENT"); fi; }

step "1. incomplete-work markers (TODO / FIXME / XXX / HACK)"
check bash -c '! grep -rnE "TODO|FIXME|XXX|HACK" src tests tools README.md CONTRIBUTING.md pyproject.toml .github --exclude=audit.sh --exclude=test_repo_hygiene.py --exclude=verify_release.py'

step "2. lint (ruff)"
check ruff check src tests tools benchmarks examples

step "3. formatting (ruff format)"
check ruff format --check src tests tools benchmarks examples

step "4. spelling (codespell)"
check codespell src tests tools benchmarks examples docs security release README.md CHANGELOG.md CONTRIBUTING.md SECURITY.md PUBLISHING.md NOTICE V2_RELEASE.txt

step "5. static types (mypy --strict)"
check mypy

step "6. security (bandit: SAST on the shipped package)"
check bandit -q -r src

step "7. secrets scan (tracked files)"
check bash -c 'n=$(git ls-files | xargs detect-secrets scan 2>/dev/null | python -c "import json,sys; print(sum(len(v) for v in json.load(sys.stdin)[\"results\"].values()))"); echo "   findings: $n"; [ "$n" = 0 ]'

step "8. dead code (vulture)"
check vulture src --min-confidence 80

step "9. tests + 100% line/branch coverage gate"
check python -m pytest -q --cov --cov-report=term-missing:skip-covered -p no:cacheprovider

step "9b. release consistency (files, versions, links, images, config, wording)"
check python tools/verify_release.py

step "10. API docs build (pdoc)"
check bash -c 'rm -rf /tmp/_apidocs && python -m pdoc -o /tmp/_apidocs munkres >/dev/null 2>/tmp/_pdoc.err && ! grep -qi "warn\|error" /tmp/_pdoc.err'

step "11. build sdist + wheel, metadata, wheel contents"
rm -rf /tmp/_dist
check bash -c 'python -m build --outdir /tmp/_dist . >/tmp/_build.log 2>&1 && ! grep -qiE "warning.*(deprecat|did not match)|Pattern .* did not match" /tmp/_build.log'
check python -m twine check /tmp/_dist/*
check check-wheel-contents /tmp/_dist/*.whl
check bash -c 'unzip -p /tmp/_dist/*.whl "*/METADATA" | grep -q "^Requires-Python: >=3.10" && ! unzip -p /tmp/_dist/*.whl "*/METADATA" | grep -q "^Requires-Dist"; echo "   python_requires set, zero runtime deps"'
check bash -c '! ls /tmp/_dist/*.whl | grep -q py2'

step "12. smoke test: install the built wheel into a clean venv, run outside the repo"
rm -rf /tmp/_smoke
check bash -c 'python -m venv /tmp/_smoke && /tmp/_smoke/bin/pip install -q /tmp/_dist/*.whl && cd /tmp && /tmp/_smoke/bin/python -m munkres >/dev/null && /tmp/_smoke/bin/python -c "
import munkres, pathlib
assert munkres.__file__.startswith(\"/tmp/_smoke\"), munkres.__file__
assert munkres.Munkres().compute([[4,1,3],[2,0,5],[3,2,2]]) == [(0,1),(1,0),(2,2)]
assert (pathlib.Path(munkres.__file__).parent / \"py.typed\").exists()
print(\"   installed\", munkres.__version__, \"- imports, solves, ships py.typed\")" && printf "4,1\n2,0\n" | /tmp/_smoke/bin/munkres - --json | grep -q "\"total\": 3" && echo "   console script \`munkres\` works"'

if [ $QUICK = 0 ]; then
  step "13. full test-suite on every supported Python (via uv)"
  for v in 3.10 3.11 3.12 3.13 3.14; do
    check bash -c "uv run --quiet --python $v --no-project --with pytest --with pytest-timeout --with hypothesis --with numpy --with scipy --with pandas --with tomli python -m pytest -q -p no:cacheprovider -x 2>&1 | tail -2 | sed 's/^/   py$v: /'; exit \${PIPESTATUS[0]}"
  done
fi

printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then printf '\033[32mAUDIT PASSED\033[0m\n'; else printf '\033[31mAUDIT FAILED (%d):\033[0m\n' ${#FAILED[@]}; printf '  - %s\n' "${FAILED[@]}"; exit 1; fi
