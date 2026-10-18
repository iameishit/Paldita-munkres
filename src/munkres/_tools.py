# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Command-line tools: doctor, inspect, certify, verify."""

from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from typing import Any

__all__ = ["run_tool"]

TOOLS = {"doctor", "inspect", "certify", "verify"}


def _doctor(args: list[str]) -> int:
    from munkres import __version__, _native

    info: dict[str, Any] = {
        "paldita_munkres": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "native_kernel": _native.version(),
    }
    for mod in ("numpy", "scipy", "pandas"):
        try:
            info[mod] = importlib.import_module(mod).__version__
        except ImportError:
            info[mod] = None
    if info["numpy"] and _native.available():
        import numpy as np

        from munkres._dense import fast_lsa

        a = np.random.default_rng(0).random((100, 100))
        fast_lsa(a)
        t = time.perf_counter()
        for _ in range(20):
            fast_lsa(a)
        info["selftest_100x100_ms"] = round((time.perf_counter() - t) / 20 * 1e3, 3)
    info["status"] = "READY (native)" if _native.available() else "READY (pure Python)"
    if "--json" in args:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:>22}: {v}")
    return 0


def _load(path: str) -> list[list[Any]]:
    from munkres.cli import _read, parse_matrix

    return parse_matrix(sys.stdin.read() if path == "-" else _read(path))


def run_tool(name: str, args: list[str]) -> int:  # noqa: PLR0911
    if name == "doctor":
        return _doctor(args)
    if not args:
        print(f"usage: munkres {name} FILE [--maximize] [--json]", file=sys.stderr)
        return 2
    from munkres._core import UnsolvableMatrix

    maximize = "--maximize" in args
    path = next(a for a in args if not a.startswith("--") or a == "-")
    try:
        matrix = _load(path)
        if name == "inspect":
            from munkres._plan import plan

            p = plan(matrix, maximize)
            fp = p.fingerprint
            if "--json" in args:
                d = {
                    k: (v if not isinstance(v, tuple) else list(v)) for k, v in fp.__dict__.items()
                }
                d["strategy"], d["reasons"] = p.strategy, list(p.reasons)
                print(json.dumps(d, default=str, indent=2))
            else:
                print(
                    f"shape {fp.shape[0]}x{fp.shape[1]}  density {fp.density:.3f}  components {fp.components}"
                )
                print(f"min {fp.min_cost}  max {fp.max_cost}  dynamic range {fp.dynamic_range}")
                print(f"Monge {fp.monge}  inverse-Monge {fp.inverse_monge}  additive {fp.additive}")
                print(
                    f"duplicate rows {fp.duplicate_rows}  duplicate columns {fp.duplicate_columns}"
                )
                print(p.explain())
            return 0
        if name == "certify":
            from munkres._plan import solve_planned

            res = solve_planned(matrix, maximize, verify=True)
            if res.certificate is None:
                print("error: nothing to certify (empty matrix)", file=sys.stderr)
                return 2
            print(res.certificate.to_json())
            return 0
        if name == "verify":  # munkres verify FILE --certificate CERT.json
            from munkres._verify import Certificate

            cpath = args[args.index("--certificate") + 1]
            with open(cpath, encoding="utf-8") as fh:
                cert = Certificate.from_json(fh.read())
            out = cert.verify(matrix)
            print(("VERIFIED: " if out else "FAILED: ") + out.reason)
            return 0 if out else 1
    except UnsolvableMatrix as bad:
        print(f"error: {bad}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, IndexError) as bad:
        print(f"error: {bad}", file=sys.stderr)
        return 2
    return 2
