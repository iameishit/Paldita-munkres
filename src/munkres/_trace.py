# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""A step-by-step record of how the solver reached its answer."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Trace"]


@dataclass
class Trace:
    """
    The solver's decisions, in order. Obtain one with `solve(matrix, trace=True)`.

    `events` holds plain dictionaries (``{"event": "star", "row": 0, "col": 1}``)
    in *solver coordinates*. The solver always works with at most as many rows as
    columns, so for a matrix with more rows than columns it works on the
    transpose (`transposed` is then true) and "row"/"column" in `events` are your
    matrix's columns/rows. `to_text` and `to_html` already translate this into
    plain language, and `to_html` produces a static page you can open in a
    browser or share.

    With gating (`max_cost` / `min_profit`) the solver also sees extra "leave
    unmatched" columns, numbered from your matrix's width upwards.
    """

    shape: tuple[int, int]
    transposed: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def _words(self) -> tuple[str, str]:
        return ("column", "row") if self.transposed else ("row", "column")

    def steps(self) -> list[str]:
        """One human-readable sentence per event."""
        row, col = self._words()
        lines: list[str] = []
        for e in self.events:
            kind = e["event"]
            if kind == "row_reduce":
                lines.append(f"Reduce {row} {e['row']} by its cheapest allowed cost, {e['min']}.")
            elif kind == "star":
                lines.append(
                    f"Match {row} {e['row']} with {col} {e['col']} (a zero after reduction)."
                )
            elif kind == "augment_start":
                lines.append(
                    f"{row.capitalize()} {e['row']} is still unmatched: search for a path."
                )
            elif kind == "adjust":
                held = e["matched_row"]
                tail = (
                    f"{col} {e['reached_col']} is free, so a path exists."
                    if held is None
                    else f"{col} {e['reached_col']} is held by {row} {held}; keep searching."
                )
                lines.append(f"Lower the reachable {col}s by {e['delta']}; {tail}")
            elif kind == "augmented":
                path = " -> ".join(str(c) for c in e["path"])
                plural = "s" if len(e["path"]) > 1 else ""
                lines.append(
                    f"Flip the path through {col}{plural} {path}: {row} {e['row']} is now matched."
                )
            else:  # "done"
                lines.append(f"Done. Pairs (row, column): {e['pairs']}.")
        return lines

    def to_text(self) -> str:
        """The whole trace as numbered plain text."""
        rows, cols = self.shape
        head = f"Hungarian algorithm on a {rows}x{cols} matrix ({len(self.events)} steps)"
        return "\n".join([head, *(f"{i:>3}. {s}" for i, s in enumerate(self.steps(), 1))])

    def to_html(self) -> str:
        """A self-contained static HTML page (no scripts, all text escaped)."""
        rows, cols = self.shape
        items = "\n".join(f"<li>{html.escape(s)}</li>" for s in self.steps())
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            "<title>munkres trace</title>"
            "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;"
            "padding:0 1rem}li{margin:.25rem 0}</style></head><body>"
            f"<h1>Hungarian algorithm on a {rows}&times;{cols} matrix</h1>"
            f"<ol>\n{items}\n</ol></body></html>\n"
        )
