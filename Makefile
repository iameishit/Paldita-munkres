.PHONY: all test audit quick dist doc clean

all: audit

test:
	python -m pytest

quick:
	tools/audit.sh --quick

audit:
	tools/audit.sh

dist:
	python -m build

doc:
	python -m pdoc -o apidocs munkres

clean:
	rm -rf dist build apidocs src/*.egg-info .pytest_cache .hypothesis .mypy_cache .ruff_cache .coverage htmlcov
