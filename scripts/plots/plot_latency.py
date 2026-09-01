#!/usr/bin/env python3
import csv
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import seaborn as sns

from sabermath.figures import (
    DEFAULT_MODEL_COLORS,
    DEFAULT_MODEL_DISPLAY_NAMES,
    DEFAULT_MODEL_MARKER_SYMBOLS,
    MODEL_ID_BY_KEY,
)
from sabermath.tables import build_rows, collect, load_timing

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results/evaluation"
TIMING_DIR = REPO / "results/timing"
OUT_DIR = REPO / "results/latency/plots"
OUT_STEM = "figure2_latency"
FALLBACK_DATA = REPO / "results/latency/data.csv"

# Colour of the 27 models with no legend entry.
OTHER_COLOR = "#C0C4C8"

FRONTIER_STYLES = {
    "bi-encoder": {"color": "#263238", "dashes": (), "linewidth": 2.6},
    "cross": {"color": "#C62828", "dashes": (3, 1.8), "linewidth": 3.2},
}

STARRED = {"jaccard", "bm25", "lightonai/Reason-ModernColBERT"}

MARKER_ZORDER = {"Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": 5}
DEFAULT_MARKER_ZORDER = 4

FRONTIER_TITLES = {
    "bi-encoder": "Bi-encoder Frontier",
    "cross": "Cross-Encoder / Late Int. Frontier",
}

X_LIM = (0.01, 150.0)
Y_LIM = (0.25, 0.80)

def fallback_latency() -> dict[str, float]:
    """Pre-measurement latencies, by figure model id. See FALLBACK_DATA."""
    if not FALLBACK_DATA.exists():
        return {}
    with FALLBACK_DATA.open(newline="", encoding="utf-8") as f:
        return {row["model_id"]: float(row["latency_s"]) for row in csv.DictReader(f)}


def frontier_class(category: str) -> str:
    """Which frontier a model competes on."""
    return "cross" if category == "RERANK" else "bi-encoder"


def mark_frontiers(rows: list[dict]) -> None:
    for row in rows:
        klass = frontier_class(row["category"])
        dominated = any(
            other is not row
            and frontier_class(other["category"]) == klass
            and other["ndcg"] >= row["ndcg"]
            and other["latency_s"] <= row["latency_s"]
            and (
                other["ndcg"] > row["ndcg"]
                or other["latency_s"] < row["latency_s"]
            )
            for other in rows
        )
        row["frontier"] = "" if dominated else klass
        # Only frontier models get a legend entry; the rest are the grey cloud.
        row["named"] = not dominated


def load_rows() -> list[dict]:
    runs, ranks = collect(RESULTS_DIR)
    scored, pending = build_rows(runs, ranks)
    timing = load_timing(TIMING_DIR)
    fallback = fallback_latency()

    rows, no_latency, fell_back = [], [], []

    for row in scored:
        entry = row.get("statement-full")
        if entry is None:
            continue
        model_id = MODEL_ID_BY_KEY.get(row["key"])
        if model_id is None:
            continue

        measured = timing.get(row["key"])
        if measured is not None:
            latency = float(measured["median_seconds"])
        elif model_id in fallback:
            latency = fallback[model_id]
            fell_back.append(row["name"])
        else:
            no_latency.append(row["name"])
            continue

        rows.append(
            {
                "model_id": model_id,
                "key": row["key"],
                "name": row["name"],
                "category": row["category"],
                "latency_s": latency,
                "ndcg": float(entry[0]),
            }
        )

    mark_frontiers(rows)

    print(f"[+] {len(rows)} models: nDCG from {RESULTS_DIR.name}, "
          f"latency from {TIMING_DIR.name}")
    if fell_back:
        print(f"[~] no timing run for {len(fell_back)} model(s), using "
              f"{FALLBACK_DATA.name}: {', '.join(sorted(fell_back))}")
    if no_latency:
        print(f"[!] no latency at all, left out: {', '.join(sorted(no_latency))}")
    if pending:
        print(f"[~] {len(pending)} model(s) with no usable run: "
              f"{', '.join(name for _, name, _ in pending)}")

    return rows

def main() -> None:
    rows = load_rows()

    display_names = DEFAULT_MODEL_DISPLAY_NAMES
    marker_symbols = DEFAULT_MODEL_MARKER_SYMBOLS
    colors = DEFAULT_MODEL_COLORS

    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    ax.set_facecolor((0.97, 0.97, 0.97))

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
        FuncFormatter(lambda v, _: f"{v:g} s")
    )

    ax.set_yticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    ax.set_ylabel("nDCG@10 (statement–full)", fontsize=22, labelpad=12)

    ax.tick_params(axis="x", labelsize=22, pad=12)
    ax.tick_params(axis="y", labelsize=23, pad=10)

    ax.grid(which="major", alpha=0.15)
    ax.set_axisbelow(True)

    sns.despine(ax=ax, left=True, bottom=True)

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

    handles = [
        frontier_handles["bi-encoder"],
        *model_handles["bi-encoder"],
        frontier_handles["cross"],
        *model_handles["cross"],
        other_handle,
    ]

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
        handlelength=2.9,
        handletextpad=0.6,
    )
    legend.set_zorder(10)

    for frontier_handle in frontier_handles.values():
        legend.get_texts()[handles.index(frontier_handle)].set_fontweight("bold")

    legends = [legend]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ("pdf", "svg"):
        path = OUT_DIR / f"{OUT_STEM}.{suffix}"
        fig.savefig(
            path, dpi=300, bbox_inches="tight", bbox_extra_artists=legends
        )
        print(f"Wrote {path.relative_to(REPO)}")

if __name__ == "__main__":
    main()
