#!/usr/bin/env python
"""Measure munkres 2.x against SciPy (and munkres 1.1.4, if the git history is present)
and write docs/BENCHMARKS.md. Numbers are for the machine that runs this script."""

import importlib.util
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import munkres  # noqa: E402

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment as scipy_lsa
except ImportError:  # pragma: no cover
    scipy_lsa = None


def old_munkres():
    """
    Load the original munkres 1.1.4 for comparison. Use the file named by $MUNKRES_1_1_4 if set,
    otherwise download the 1.1.4 release from PyPI. Returns None (the column is skipped) if neither works.
    """
    path = os.environ.get("MUNKRES_1_1_4")
    if not path:
        scratch = tempfile.mkdtemp()
        done = subprocess.run(
            [sys.executable, "-m", "pip", "download", "munkres==1.1.4", "--no-deps", "-q", "-d", scratch],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        archives = [f for f in os.listdir(scratch) if f.endswith(".whl")]
        if done.returncode != 0 or not archives:
            return None
        with zipfile.ZipFile(os.path.join(scratch, archives[0])) as wheel:
            wheel.extract("munkres.py", scratch)
        path = os.path.join(scratch, "munkres.py")
    spec = importlib.util.spec_from_file_location("old_munkres", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def draw_chart(results):
    """Write assets/benchmark.png (needs matplotlib; returns False if it is not installed)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    navy, panel, ink = "#0f1630", "#16204a", "#e9eefc"
    colours = {"munkres 1.1.4": "#8a93b2", "munkres 2.x": "#3d8bff", "SciPy (compiled)": "#f5b942"}
    labels = [r[0] for r in results][::-1]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(navy)
    ax.set_facecolor(panel)
    height = 0.26
    for k, (series, idx) in enumerate(
        (("munkres 1.1.4", 1), ("munkres 2.x", 2), ("SciPy (compiled)", 3))
    ):
        ys, vals = [], []
        for y, row in enumerate(results[::-1]):
            if row[idx] is not None:
                ys.append(y + (1 - k) * height)
                vals.append(max(row[idx], 1e-4))
        bars = ax.barh(ys, vals, height=height * 0.92, color=colours[series], label=series)
        for bar, value in zip(bars, vals, strict=True):
            ax.text(
                value * 1.12,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f} s" if value >= 0.001 else "<0.001 s",
                va="center",
                color=ink,
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=ink, fontsize=10)
    ax.tick_params(axis="x", colors=ink)
    ax.set_xlabel("seconds (log scale, lower is better)", color=ink)
    ax.set_xlim(right=max(r[i] for r in results for i in (1, 2, 3) if r[i]) * 12)
    for spine in ax.spines.values():
        spine.set_color("#2d3a6e")
    ax.grid(axis="x", color="#2d3a6e", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "Paldita Munkres V2: solve time, munkres 1.1.4 vs 2.x vs SciPy",
        color=ink,
        fontsize=13,
        pad=14,
        loc="left",
    )
    legend = ax.legend(
        loc="upper right", facecolor=navy, edgecolor="#2d3a6e", labelcolor=ink, fontsize=9
    )
    legend.get_frame().set_alpha(0.9)
    fig.text(
        0.01,
        0.01,
        f"{platform.python_implementation()} {platform.python_version()}, one run per bar (tools/benchmark.py). munkres 1.1.4 was not measured above 300x300.",
        color="#8a93b2",
        fontsize=7,
    )
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "benchmark.png"), facecolor=navy)
    return True


def main():
    rng = random.Random(0)
    old = old_munkres()
    cases = [
        ("random int 50x50", 50, lambda n: [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]),
        ("random int 100x100", 100, lambda n: [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]),
        ("random int 200x200", 200, lambda n: [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]),
        ("random float 200x200", 200, lambda n: [[rng.random() * 1000 for _ in range(n)] for _ in range(n)]),
        ("many ties (0..3) 200x200", 200, lambda n: [[rng.randint(0, 3) for _ in range(n)] for _ in range(n)]),
        ("random int 500x500", 500, lambda n: [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]),
        ("random int 1000x1000", 1000, lambda n: [[rng.randint(0, 1000) for _ in range(n)] for _ in range(n)]),
    ]  # fmt: skip
    lines = [
        "# Benchmarks",
        "",
        f"Measured on `{platform.platform()}`, Python {platform.python_version()}, "
        f"single run per cell (`tools/benchmark.py`). Your numbers will differ; the "
        "ratios are what matter.",
        "",
        "| case | munkres 1.1.4 | **munkres 2.x** | SciPy (compiled) |",
        "|------|--------------:|----------------:|-----------------:|",
    ]
    results = []
    for name, n, make in cases:
        matrix = make(n)
        new_t = timed(munkres.Munkres().compute, matrix)
        old_t = timed(old.Munkres().compute, matrix) if old and n <= 300 else None
        sci_t = timed(scipy_lsa, np.array(matrix, dtype=float)) if scipy_lsa else None
        fmt = lambda t: "n/a" if t is None else f"{t:.3f} s"  # noqa: E731
        lines.append(f"| {name} | {fmt(old_t)} | **{fmt(new_t)}** | {fmt(sci_t)} |")
        results.append((name, old_t, new_t, sci_t))
    lines += [
        "",
        "munkres 2.x is pure Python (no dependencies); SciPy is compiled C++. Use SciPy "
        "when raw speed on big dense problems is all that matters. 1.1.4 was not "
        "measured above 300x300.",
        "",
    ]
    chart = draw_chart(results)
    if chart:
        lines.insert(4, "![Benchmark results](../assets/benchmark.png)")
        lines.insert(5, "")
    out = os.path.join(ROOT, "docs", "BENCHMARKS.md")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
