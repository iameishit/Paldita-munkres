# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Command line interface: ``munkres costs.csv --maximize --json``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from munkres import __version__
from munkres._api import solve
from munkres._core import DISALLOWED, UnsolvableMatrix

_FORBIDDEN_TOKENS = {
    "",
    "d",
    "x",
    "-",
    "none",
    "disallowed",
    "inf",
    "+inf",
    "infinity",
    "+infinity",
}


def parse_number(token: str) -> Any:
    """`int` if it looks like one, else `float`."""
    try:
        return int(token)
    except ValueError:
        return float(token)  # raises ValueError for junk


def parse_matrix(text: str) -> list[list[Any]]:
    """
    Read a matrix from comma- or whitespace-separated text. Blank lines and
    ``#`` comments are skipped. An empty cell, ``D``, ``x``, ``-`` or ``inf``
    means "forbidden".
    """
    matrix: list[list[Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue
        tokens = line.split(",") if "," in line else line.split()
        row: list[Any] = []
        for token in (t.strip() for t in tokens):
            if token.lower() in _FORBIDDEN_TOKENS:
                row.append(DISALLOWED)
                continue
            try:
                row.append(parse_number(token))
            except ValueError:
                raise ValueError(f"line {number}: cannot read {token!r} as a number") from None
        matrix.append(row)
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="munkres",
        description="Solve an assignment problem (Hungarian algorithm). With no "
        "arguments, run a built-in self-check.",
        epilog="Matrix format: one row per line, cells separated by commas or "
        "spaces; an empty cell, D, x, - or inf forbids a pairing; # starts a comment.",
    )
    parser.add_argument("file", nargs="?", help="matrix file, or - for standard input")
    parser.add_argument("--maximize", action="store_true", help="maximise instead of minimise")
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--max-cost", metavar="X", help="never make pairs costing more than X")
    gate.add_argument("--min-profit", metavar="X", help="never make pairs earning less than X")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--trace", action="store_true", help="also print the solver's steps")
    parser.add_argument("--trace-html", metavar="FILE", help="write the steps as an HTML page")
    parser.add_argument("--self-check", action="store_true", help="run the built-in self-check")
    parser.add_argument("--version", action="version", version=f"munkres {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status (0 ok, 1 unsolvable, 2 bad input)."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)

    if args.self_check or not args_list:
        from munkres.__main__ import main as self_check  # noqa: PLC0415

        self_check()
        return 0
    if args.file is None:
        print("error: give a matrix file (or - for standard input)", file=sys.stderr)
        return 2

    try:
        text = sys.stdin.read() if args.file == "-" else _read(args.file)
        matrix = parse_matrix(text)
        threshold = args.min_profit if args.maximize else args.max_cost
        wrong = args.max_cost if args.maximize else args.min_profit
        if wrong is not None:
            flag = "--max-cost" if args.maximize else "--min-profit"
            raise ValueError(f"{flag} does not apply here; use the other gate flag")
        result = solve(
            matrix,
            maximize=args.maximize,
            max_cost=None if args.maximize else _opt_number(threshold),
            min_profit=_opt_number(threshold) if args.maximize else None,
            trace=args.trace or bool(args.trace_html),
        )
        if args.trace_html:
            with open(args.trace_html, "w", encoding="utf-8") as handle:
                handle.write(result.trace.to_html() if result.trace is not None else "")
    except UnsolvableMatrix as bad:
        print(f"error: {bad}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as bad:
        print(f"error: {bad}", file=sys.stderr)
        return 2

    _print(result, matrix, as_json=args.json, show_trace=args.trace)
    return 0


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _opt_number(token: str | None) -> Any:
    return None if token is None else parse_number(token)


def _print(result: Any, matrix: list[list[Any]], *, as_json: bool, show_trace: bool) -> None:
    if as_json:
        payload: dict[str, Any] = {
            "pairs": [list(p) for p in result.pairs],
            "total": result.total,
            "unmatched_rows": list(result.unmatched_rows),
            "unmatched_cols": list(result.unmatched_cols),
            "shape": list(result.shape),
            "maximize": result.maximize,
        }
        if show_trace and result.trace is not None:
            payload["trace"] = result.trace.steps()
        print(json.dumps(payload))
        return
    for r, c in result.pairs:
        print(f"{r} {c} {matrix[r][c]}")
    print(f"total: {result.total}")
    if result.unmatched_rows:
        print(f"unmatched rows: {list(result.unmatched_rows)}")
    if result.unmatched_cols:
        print(f"unmatched columns: {list(result.unmatched_cols)}")
    if show_trace and result.trace is not None:
        print(result.trace.to_text())
