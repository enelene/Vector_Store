#!/usr/bin/env python3
"""Generate academic-style benchmark figures for the VectorDB thesis.

Reads a long-format CSV (see ``benchmark_results.csv``) in which an
``experiment`` column selects the sweep, ``x_value`` is the swept parameter, and
the remaining columns are metrics (blank cells are simply ignored). It emits
clean, 300-DPI PNGs ready to drop into a thesis or paper:

    recall_vs_efsearch.png   Recall@1 / Recall@10 vs. the search beam width (ef)
    qps_vs_recall.png        Throughput-accuracy trade-off (the classic ANN curve)
    scaling_vs_nodes.png     Read vs. write throughput as the cluster grows
    latency_vs_efsearch.png  Query latency (p50/p99) vs. the search beam width

Usage:
    python scripts/generate_plots.py [--csv benchmark_results.csv] [--outdir docs/figures]

Dependencies: pandas, matplotlib.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed (CI / servers)

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import pandas as pd  # noqa: E402

# Colour-blind-friendly palette (Wong, 2011).
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"


def set_academic_style() -> None:
    """Apply a clean, paper-ready Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 120,
            "savefig.dpi": 300,          # high-res for print
            "savefig.bbox": "tight",
            "font.family": "serif",      # matches most thesis body text
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _save(fig: "plt.Figure", outdir: str, name: str) -> None:
    """Write a figure to ``outdir/name`` and report the path."""
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_recall_vs_efsearch(df: pd.DataFrame, outdir: str) -> None:
    """Recall@1 and Recall@10 as a function of the search beam width ef."""
    d = df[df["experiment"] == "efsearch"].sort_values("x_value")
    if d.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(d["x_value"], d["recall_at_1"], marker="o", color=BLUE, label="Recall@1")
    ax.plot(d["x_value"], d["recall_at_10"], marker="s", color=ORANGE, label="Recall@10")
    ax.set_xlabel("Search beam width  $ef_{search}$")
    ax.set_ylabel("Recall")
    ax.set_title("Accuracy vs. Search Beam Width")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    _save(fig, outdir, "recall_vs_efsearch.png")


def plot_qps_vs_recall(df: pd.DataFrame, outdir: str) -> None:
    """Throughput-accuracy trade-off: QPS against Recall@10 (Pareto curve)."""
    d = df[df["experiment"] == "efsearch"].sort_values("recall_at_10")
    if d.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(d["recall_at_10"], d["qps"], marker="o", color=GREEN)
    # Annotate each operating point with its ef value.
    for _, row in d.iterrows():
        ax.annotate(
            f"ef={int(row['x_value'])}",
            (row["recall_at_10"], row["qps"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Search throughput (queries/sec)")
    ax.set_title("Throughput-Accuracy Trade-off")
    _save(fig, outdir, "qps_vs_recall.png")


def plot_scaling_vs_nodes(df: pd.DataFrame, outdir: str) -> None:
    """Read vs. write throughput as the cluster grows (dual y-axis).

    Demonstrates the architectural property: reads are served locally so they
    scale with the cluster, while full replication keeps write throughput flat.
    """
    d = df[df["experiment"] == "nodecount"].sort_values("x_value")
    if d.empty:
        return
    fig, ax1 = plt.subplots()
    ax1.plot(d["x_value"], d["qps"], marker="o", color=BLUE,
             label="Search throughput (QPS)")
    ax1.set_xlabel("Cluster size (nodes)")
    ax1.set_ylabel("Search throughput (queries/sec)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(d["x_value"].tolist())

    ax2 = ax1.twinx()
    ax2.grid(False)  # avoid a second, clashing grid
    ax2.plot(d["x_value"], d["ingestion_per_sec"], marker="s", color=ORANGE,
             label="Ingestion (inserts/sec)")
    ax2.set_ylabel("Ingestion (inserts/sec)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0, max(d["ingestion_per_sec"]) * 1.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Cluster Scaling: Reads Scale, Writes Do Not")
    _save(fig, outdir, "scaling_vs_nodes.png")


def plot_latency_vs_efsearch(df: pd.DataFrame, outdir: str) -> None:
    """Query latency percentiles (p50, p99) vs. the search beam width ef."""
    d = df[df["experiment"] == "efsearch"].sort_values("x_value")
    if d.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(d["x_value"], d["latency_p50_ms"], marker="o", color=BLUE, label="p50")
    ax.plot(d["x_value"], d["latency_p99_ms"], marker="s", color=PINK, label="p99")
    ax.set_xlabel("Search beam width  $ef_{search}$")
    ax.set_ylabel("Query latency (ms)")
    ax.set_title("Query Latency vs. Search Beam Width")
    ax.legend(loc="upper left")
    _save(fig, outdir, "latency_vs_efsearch.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="benchmark_results.csv",
                        help="input CSV (default: benchmark_results.csv)")
    parser.add_argument("--outdir", default=os.path.join("docs", "figures"),
                        help="output directory for PNGs (default: docs/figures)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    set_academic_style()

    print(f"Reading {args.csv}; writing figures to {args.outdir}/")
    plot_recall_vs_efsearch(df, args.outdir)
    plot_qps_vs_recall(df, args.outdir)
    plot_scaling_vs_nodes(df, args.outdir)
    plot_latency_vs_efsearch(df, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
