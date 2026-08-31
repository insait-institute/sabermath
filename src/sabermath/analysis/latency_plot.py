"""
Latency vs. retrieval quality on SABER-Math (statement-full), all 42 models.

A matplotlib rebuild of ``figure2-latency-mini.svg``, drawn in the style of
the math-vs-word figure so the two can be read as one pair:

  * the axes are the size the SELECTED math-vs-word figure uses (15x8 at the
    same tick/label sizes and the same s=120 markers) - only the legend
    departs from it, sitting below the axes as on the --all figure rather
    than in-plot, because 15 entries in-plot would cover the frontier;
  * every model that carries a legend entry is drawn with the marker and
    colour ``plot_hist.py`` gives it, and labelled with the name it uses
    there - read live out of that file (see plot_hist_registry), never
    copied, so the two figures cannot drift apart;
  * the other 27 models stay anonymous grey circles, exactly as in the mini.

    python experiments/latency/plot_latency.py

Data comes from ``data.csv``; see ``extract_from_svg.py`` for where each of
its two columns is sourced.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

import plot_hist_registry

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.csv"
OUT_DIR = HERE / "plots"
OUT_STEM = "figure2_latency"

# Colour of the 27 models with no legend entry.
OTHER_COLOR = "#C0C4C8"

# The two frontiers differ in hue, dash and weight, so which line is which
# reads without going to the legend.
#
# The cross-encoder red is a DEEPER red than plot_hist.py's #e53935, which is
# RaDeR-7B's marker colour. That matters here: RaDeR-7B sits at (1.67s, 0.690)
# and this line runs along y=0.693 from 1.17s out to 63.9s, so the two pass
# within about five pixels of each other. Deepening the line keeps the marker
# readable where they nearly touch. #c62828 is registered to Rank1-32B, which
# is one of the anonymous grey circles here and so never draws it.
FRONTIER_STYLES = {
    "bi-encoder": {"color": "#263238", "dashes": (), "linewidth": 2.6},
    "cross": {"color": "#C62828", "dashes": (3, 1.8), "linewidth": 3.2},
}

# Carried over verbatim from the mini figure, whose caption defines what the
# mark means. Kept rather than dropped: the caption refers to it.
STARRED = {"jaccard", "bm25", "lightonai/Reason-ModernColBERT"}

# RaDeR-3B and Qwen3-Reranker-0.6B are 0.03 s and 0.002 nDCG apart - close
# enough that whichever is drawn second hides the other. They are on different
# frontiers, so draw order would otherwise be decided by which frontier the
# loop below reaches first; this pins it.
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

    display_names = plot_hist_registry.display_names()
    marker_symbols = plot_hist_registry.marker_symbols()
    colors = plot_hist_registry.colors()

    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    ax.set_facecolor((0.97, 0.97, 0.97))

    # -----------------------------------------------------------------
    # The 27 models with no legend entry: one grey cloud, one handle.
    # Slightly smaller and below the named markers, so a frontier model
    # landing on top of one still reads as the frontier model.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # The two frontier lines, drawn under the markers.
    #
    # steps-pre reproduces the mini's staircase exactly: from each model the
    # line rises to the NEXT model's score first and only then runs right to
    # its latency, so the step meets every marker it names.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # The 15 named models, in each frontier's own left-to-right order -
    # which is also the order their legend column reads in.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Axes.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Legend: one box under the axes, three columns, each frontier group
    # headed by its own line in bold.
    # -----------------------------------------------------------------
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

    # One flat run: each frontier line, bolded, heads the group it belongs to.
    # matplotlib fills columns top-to-bottom, so at 3 columns the 18 entries
    # divide exactly 6/6/6 and the two headings simply fall where they fall -
    # a group is allowed to continue into the next column. Giving each group
    # a column of its own instead would set the legend's height to the taller
    # group's 10 rows, which is the vertical space this is spending 3 columns
    # to avoid.
    handles = [
        frontier_handles["bi-encoder"],
        *model_handles["bi-encoder"],
        frontier_handles["cross"],
        *model_handles["cross"],
        other_handle,
    ]

    # The box spans the figure's full visual width: its left edge lines up
    # with the top of the rotated y axis label - which, rotated, is that
    # label's leftmost extent - and its right edge with the right edge of the
    # plot. The label's position is only known once the text has been laid
    # out, so draw first and measure rather than guessing an offset, and let
    # mode="expand" stretch the three columns across whatever that leaves.
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
        # 43.5pt at this font size. matplotlib scales a dash pattern by the
        # line's width, so the cross-encoder frontier's (3, 1.8) at linewidth
        # 3.2 has a period of 15.4pt and its third dash ends at 40.3pt: this
        # handle closes in the gap after it, rather than a hair into a fourth
        # dash the way 3.2 (48pt) did. The default handlelength=2.0 would be
        # 30pt, and much above 3 the entries stop fitting the width
        # mode="expand" pins them to and the labels run into the next
        # column's handle.
        handlelength=2.9,
        handletextpad=0.6,
    )
    legend.set_zorder(10)

    for frontier_handle in frontier_handles.values():
        legend.get_texts()[handles.index(frontier_handle)].set_fontweight("bold")

    legends = [legend]

    # NOT tight_layout: like the --all math-vs-word figure, it would grow the
    # axes box down into the gap the legends are anchored in. bbox_inches
    # below extends the saved canvas around them instead.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ("pdf", "svg"):
        path = OUT_DIR / f"{OUT_STEM}.{suffix}"
        # bbox_extra_artists keeps the legend in frame: a tight bbox does
        # not measure an artist anchored outside the axes on its own.
        fig.savefig(
            path, dpi=300, bbox_inches="tight", bbox_extra_artists=legends
        )
        print(f"Wrote {path.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
