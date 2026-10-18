# Development

```bash
git clone https://github.com/iameishit/Paldita-munkres && cd Paldita-munkres
uv sync                     # or: python -m venv .venv && pip install -e . --group dev
python -m pytest            # tests
tools/audit.sh --quick      # lint, format, spelling, types, security, coverage, build, smoke test
tools/audit.sh              # ... plus every supported Python (needs uv)
```

Layout: `src/munkres/` (library), `tests/` (suites by area), `tools/` (scripts), `benchmarks/`, `examples/`,
`docs/`. Conventions: tests first, independent oracles for new algorithms, 100 % coverage, `mypy --strict`,
no runtime dependencies. See [CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENTS.md](../AGENTS.md).

Handy: `just --list`, `pre-commit install`, the dev container in `.devcontainer/`.
