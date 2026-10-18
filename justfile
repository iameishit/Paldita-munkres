# `just --list` shows every recipe.
default:
    @just --list

install:
    uv sync

test:
    python -m pytest

cov:
    python -m pytest --cov

lint:
    ruff check src tests tools benchmarks examples
    ruff format --check src tests tools benchmarks examples

fmt:
    ruff format src tests tools benchmarks examples
    ruff check --fix src tests tools benchmarks examples

typecheck:
    mypy

quick:
    tools/audit.sh --quick

audit:
    tools/audit.sh

build:
    python -m build

docs:
    python -m pdoc -o apidocs munkres

bench:
    python tools/benchmark.py

verify-package:
    python tools/verify_package.py

verify-release:
    python tools/verify_release.py

clean:
    rm -rf dist build apidocs src/*.egg-info .pytest_cache .hypothesis .mypy_cache .ruff_cache .coverage htmlcov
