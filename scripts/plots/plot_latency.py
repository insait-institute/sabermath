#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from sabermath.figures import (
    DEFAULT_MODEL_COLORS,
    DEFAULT_MODEL_DISPLAY_NAMES,
    DEFAULT_MODEL_MARKER_SYMBOLS,
)

# parents: [0] analysis, [1] sabermath, [2] src, [3] the repo root.
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "results/latency/data.csv"
OUT_DIR = REPO / "results/latency/plots"
OUT_STEM = "figure2_latency"

# Colour of the 27 models with no legend entry.
OTHER_COLOR = "#C0C4C8"

# The two frontiers differ in hue, dash and weight, so which is which reads
# without the legend. The cross-encoder red is deeper than plot_hist's #e53935
# because that is RaDeR-7B's marker colour and the line passes within a few
# pixels of it.
FRONTIER_STYLES = {
    "bi-encoder": {"color": "#263238", "dashes": (), "linewidth": 2.6},
    "cross": {"color": "#C62828", "dashes": (3, 1.8), "linewidth": 3.2},
}

# From the mini figure, whose caption defines what the mark means.
STARRED = {"jaccard", "bm25", "lightonai/Reason-ModernColBERT"}

# These two are 0.03 s and 0.002 nDCG apart, close enough that whichever draws
# second hides the other. They sit on different frontiers, so pin the order.
MARKER_ZORDER = {"Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": 5}
DEFAULT_MARKER_ZORDER = 4

FRONTIER_TITLES = {
    "bi-encoder": "Bi-encoder Frontier",
    "cross": "Cross-Encoder / Late Int. Frontier",
}

# Axis window, matching the mini's: one decade every 75.7px from 0.01s, and
# 0.25-0.80 nDCG.
X_LIM = (0.01, 150.0)
Y_LIM = (0.25, 0.80)


def load_rows() -> list[dict]:
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["latency_s"] = float(row["latency_s"])
        row["ndcg"] = float(row["ndcg"])
        row["named"] = row["named"] == "1"

    return rows


def main() -> None:
    rows = load_rows()

    display_names = DEFAULT_MODEL_DISPLAY_NAMES
    marker_symbols = DEFAULT_MODEL_MARKER_SYMBOLS
    colors = DEFAULT_MODEL_COLORS

    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    ax.set_facecolor((0.97, 0.97, 0.97))

    # The unnamed models: one grey cloud, one handle. Drawn smaller and below
    # the named markers, so a frontier model landing on one still reads.
    others = [row for row in rows if not row["named"]]

    ax.scatter(
        [row["latency_s"] for row in others],
        [row["ndcg"] for row in others],
        color=OTHER_COLOR,
        marker="o",
        s=100,
        alpha=0.75,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    # steps-pre reproduces the mini's staircase: the line rises to the next
    # model's score before running right to its latency, so each step meets the
    # marker it names.
    frontier_handles = {}

    for frontier, style in FRONTIER_STYLES.items():
        points = [row for row in rows if row["frontier"] == frontier]
        points.sort(key=lambda row: row["latency_s"])

        (line,) = ax.plot(
            [row["latency_s"] for row in points],
            [row["ndcg"] for row in points],
            drawstyle="steps-pre",
            alpha=0.9,
            zorder=2,
            **style,
        )
        line.set_label(FRONTIER_TITLES[frontier])
        frontier_handles[frontier] = line

    # In each frontier's left-to-right order, which is also its legend order.
    model_handles = {"bi-encoder": [], "cross": []}

    for frontier in ("bi-encoder", "cross"):
        points = [row for row in rows if row["frontier"] == frontier]
        points.sort(key=lambda row: row["latency_s"])

        for row in points:
            model_id = row["model_id"]
            label = display_names[model_id]

            if model_id in STARRED:
                label += "*"

            handle = ax.scatter(
                row["latency_s"],
                row["ndcg"],
                label=label,
                color=colors[model_id],
                marker=marker_symbols[model_id],
                s=120,
                edgecolors="none",
                linewidths=0,
                zorder=MARKER_ZORDER.get(model_id, DEFAULT_MARKER_ZORDER),
            )
            model_handles[frontier].append(handle)

    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0, 100.0]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(
        # "0.01 s" .. "100 s" - the unit rides on the tick, as in the mini,
        # which is why there is no x label.
        FuncFormatter(lambda v, _: f"{v:g} s")
    )

    ax.set_yticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    ax.set_ylabel("nDCG@10 (statement–full)", fontsize=22, labelpad=12)

    # Padded outwards: at the origin the bottom y label and the leftmost x
    # label are diagonal neighbours and sat almost touching.
    ax.tick_params(axis="x", labelsize=22, pad=12)
    ax.tick_params(axis="y", labelsize=23, pad=10)

    ax.grid(which="major", alpha=0.15)
    ax.set_axisbelow(True)

    sns.despine(ax=ax, left=True, bottom=True)

    # One box under the axes, three columns, each frontier group headed by its
    # own line in bold.
    other_handle = Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=9,
        markerfacecolor=OTHER_COLOR,
        markeredgecolor="none",
        alpha=0.75,
        label="All other models",
    )

    # One flat run rather than a column per group: matplotlib fills columns
    # top-to-bottom, and a column per group would set the legend's height to
    # the taller group's row count - the space these 3 columns exist to save.
    handles = [
        frontier_handles["bi-encoder"],
        *model_handles["bi-encoder"],
        frontier_handles["cross"],
        *model_handles["cross"],
        other_handle,
    ]

    # Spans the figure's full visual width, from the rotated y label's leftmost
    # extent to the right edge of the plot. That position is only known once
    # the text is laid out, so draw first and measure rather than guess.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_left = ax.yaxis.label.get_window_extent(renderer).x0
    legend_x = ax.transAxes.inverted().transform((label_left, 0.0))[0]

    legend = ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper left",
        bbox_to_anchor=(legend_x, -0.10, 1.0 - legend_x, 0.0),
        mode="expand",
        bbox_transform=ax.transAxes,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=15,
        ncol=3,
        borderaxespad=0.0,
        labelspacing=0.55,
        columnspacing=2.0,
        # matplotlib scales a dash pattern by line width, so this length ends
        # the handle in the gap after a dash rather than partway into one.
        # Much above 3 and the labels run into the next column's handle.
        handlelength=2.9,
        handletextpad=0.6,
    )
    legend.set_zorder(10)

    for frontier_handle in frontier_handles.values():
        legend.get_texts()[handles.index(frontier_handle)].set_fontweight("bold")

    legends = [legend]

    # Not tight_layout: it would grow the axes box down into the gap the
    # legends are anchored in. bbox_inches extends the canvas instead.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ("pdf", "svg"):
        path = OUT_DIR / f"{OUT_STEM}.{suffix}"
        # A tight bbox does not measure an artist anchored outside the axes.
        fig.savefig(
            path, dpi=300, bbox_inches="tight", bbox_extra_artists=legends
        )
        print(f"Wrote {path.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
