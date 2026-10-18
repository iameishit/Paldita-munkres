# Release notes: munkres 3.0.0

The first stable 3.x release. The full notes are in [V3_RELEASE.txt](../V3_RELEASE.txt); the complete
list of changes is in [CHANGELOG.md](../CHANGELOG.md).

## Highlights

- Optional native C kernel with automatic, verified fallback to pure Python
- Exact certificates and an independent verifier
- A planner that reports which algorithm it chose and why
- Dynamic repair after cost changes, plus structure, sensitivity and risk tools
- On NumPy input, about the same speed as SciPy in the measured range (0.75x-1.04x), same optimum

## Upgrading

Code written for 2.0.0 needs no changes. See [docs/migration.md](../docs/migration.md).
