"""
Recover the underlying data of ``figure2-latency-mini.svg`` into ``data.csv``.

The mini figure was authored as hand-written SVG, outside this repo, and the
per-query LATENCIES it plots exist nowhere else here: ``results/timing/`` holds
only 7 of the 42 models. The SVG is therefore the sole source for the x axis,
and this script is what makes it a reproducible one - it inverts the figure's
own axis transforms rather than re-typing numbers off a picture.

The y axis needs no such recovery. ``experiments/rescaling_robustness/
results.json`` already carries the authoritative statement-full nDCG@10 for
exactly these 42 models, so every marker is matched back to a row there and
the CSV stores the REPOSITORY's number, not the pixel-derived one (which
agrees to <=0.004 for all 42 - see --verify).

Matching is not done by nDCG alone: two pairs of models are closer together
than the pixel error (Qwen3-Reranker-0.6B/SPLADE-Code-8B, Jaccard/TF-IDF), and
a naive nearest-value assignment swaps both. The 15 markers that carry a
legend entry are pinned first by their (fill, shape) - which the legend
defines unambiguously - and only the 27 anonymous grey circles are then
matched by rank against whatever rows are left.

    python experiments/latency/extract_from_svg.py [--verify]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import plot_hist_registry

REPO = Path(__file__).resolve().parents[2]
SVG = REPO / "figure2-latency-mini.svg"
RESCALING = REPO / "experiments/rescaling_robustness/results.json"
OUT = Path(__file__).resolve().parent / "data.csv"

# --- Axis calibration, read off the SVG's own gridlines ---------------
# x: log10 latency in seconds. The "0.01 s" .. "100 s" gridlines sit at
# x=36.0 .. 338.7, i.e. one decade every (338.7-36.0)/4 px.
X_AT_1E_MINUS_2, PX_PER_DECADE = 36.0, (338.7 - 36.0) / 4.0
# y: linear nDCG. The "0.25" gridline is at y=282.0, the "0.75" one at 34.7.
Y_AT_0_25, PX_PER_NDCG = 282.0, (282.0 - 34.7) / 0.5


def to_latency(x_px: float) -> float:
    return 10.0 ** ((x_px - X_AT_1E_MINUS_2) / PX_PER_DECADE - 2.0)


def to_ndcg(y_px: float) -> float:
    return 0.25 + (Y_AT_0_25 - y_px) / PX_PER_NDCG


# --- The 15 legend entries, keyed by the marker's (fill, shape) -------
# Values are ids from experiments/math-vs-word/plot_hist.py, which owns the
# display name, marker and colour each of these is drawn with. The SVG's own
# fills and shapes are NOT reused - only its geometry is.
LEGEND = {
    ("#4B4848", "poly3"): "jaccard",
    ("#5F6368", "plus"): "bm25",
    ("#8D6E63", "poly3"): "BAAI/bge-m3",
    ("#00C2A8", "poly4"): "Qwen/Qwen3-Embedding-0.6B",
    ("#00A651", "poly4"): "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    ("#1565C0", "circle"): "AQ-MedAI/Diver-Retriever-4B",
    ("#00A651", "rect"): "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    ("#D500F9", "rect"): "hanhainebula/reason-embed-qwen3-8b-0928",
    ("#EA80FC", "circle"): "reason-rewriter-reason-embed-8b",
    ("#E53935", "rect"): "lightonai/Reason-ModernColBERT",
    ("#00C2A8", "poly6"): "qwen3-reranker-0.6b",
    ("#00C2A8", "poly3"): "qwen3-reranker-4b",
    ("#1565C0", "poly4"): "diver-grouprank-32b",
    ("#6A00FF", "poly6"): "retro-star-32b",
    ("#FF4FD8", "poly3"): "retro-star-32b-rewritten",
}

# Which frontier line each legend model sits on, in the order the SVG's two
# step paths visit them. Everything else is an anonymous grey circle.
BI_ENCODER_FRONTIER = [
    "jaccard",
    "bm25",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "AQ-MedAI/Diver-Retriever-4B",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "hanhainebula/reason-embed-qwen3-8b-0928",
    "reason-rewriter-reason-embed-8b",
]
CROSS_FRONTIER = [
    "lightonai/Reason-ModernColBERT",
    "qwen3-reranker-0.6b",
    "qwen3-reranker-4b",
    "diver-grouprank-32b",
    "retro-star-32b",
    "retro-star-32b-rewritten",
]

# The rescaling table names models the way the paper's main table does. Two of
# its rows are named differently in plot_hist.py (which spells the composed
# systems out as "ReasonRewriter + <scorer>"); the rest match verbatim.
RESCALING_NAME_ALIASES = {
    "ReasonEmbed-Qwen3-8B-Rewrite": "reason-rewriter-reason-embed-8b",
    "ReasonReranker-Qwen3-32B-Rewrite": "retro-star-32b-rewritten",
}


def parse_markers(svg_text: str) -> list[dict]:
    """Every data marker in the plot area, as (x_px, y_px, fill, shape)."""
    # Stop at the legend box - its swatches are markers too.
    plot = svg_text.split('<rect x="36.0" y="301.0"')[0]
    markers = []

    for m in re.finditer(
        r'<circle cx="([\d.]+)" cy="([\d.]+)" r="[\d.]+" fill="(#\w+)"([^>]*)>', plot
    ):
        markers.append(
            {
                "x": float(m[1]),
                "y": float(m[2]),
                "fill": m[3],
                "shape": "circle",
                # The 27 anonymous models are the only markers drawn with
                # opacity; the legend calls them "All other models".
                "anonymous": "opacity" in m[4],
            }
        )

    for m in re.finditer(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="(#\w+)"',
        plot,
    ):
        if m[5] in ("#FFFFFF", "#F7F7F7"):  # canvas and axes background
            continue
        markers.append(
            {
                "x": float(m[1]) + float(m[3]) / 2,
                "y": float(m[2]) + float(m[4]) / 2,
                "fill": m[5],
                "shape": "rect",
                "anonymous": False,
            }
        )

    for m in re.finditer(r'<polygon points="([^"]+)" fill="(#\w+)"', plot):
        xs, ys = zip(*(tuple(map(float, p.split(","))) for p in m[1].split()))
        markers.append(
            {
                # A triangle's centroid is not its centre: matplotlib centres
                # the marker's bounding box, so use that for every polygon.
                "x": sum(xs) / 3 if len(xs) == 3 else (min(xs) + max(xs)) / 2,
                "y": (min(ys) + max(ys)) / 2,
                "fill": m[2],
                "shape": f"poly{len(xs)}",
                "anonymous": False,
            }
        )

    # BM25's plus is a stroked path, not a filled shape.
    for m in re.finditer(
        r'<path d="M [\d.]+ ([\d.]+) H [\d.]+ M ([\d.]+) [\d.]+ V [\d.]+" stroke="(#\w+)"',
        plot,
    ):
        markers.append(
            {
                "x": float(m[2]),
                "y": float(m[1]),
                "fill": m[3],
                "shape": "plus",
                "anonymous": False,
            }
        )

    return markers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Print the pixel-derived nDCG next to the repository's own value "
        "for every model, and the largest disagreement.",
    )
    args = ap.parse_args()

    markers = parse_markers(SVG.read_text(encoding="utf-8"))

    rescaling = json.loads(RESCALING.read_text(encoding="utf-8"))

    # plot_hist.py keys models by their HF repo id; the rescaling table keys
    # them by the display name the paper's main table uses. Join the two
    # through those names, so a name neither side knows is an error rather
    # than a silently mislabelled point.
    display_names = plot_hist_registry.display_names()
    id_by_display_name = {name: mid for mid, name in display_names.items()}

    ndcg_by_id = {}
    for name, value in zip(rescaling["models"], rescaling["table"]):
        model_id = RESCALING_NAME_ALIASES.get(name) or id_by_display_name.get(name)
        if model_id is None:
            raise SystemExit(
                f"{name!r} from {RESCALING.name} has no id in plot_hist.py's "
                "DEFAULT_MODEL_DISPLAY_NAMES - add it there or to "
                "RESCALING_NAME_ALIASES."
            )
        ndcg_by_id[model_id] = value

    named = [m for m in markers if not m["anonymous"]]
    anonymous = [m for m in markers if m["anonymous"]]

    if len(markers) != len(ndcg_by_id):
        raise SystemExit(
            f"{len(markers)} markers in the SVG but {len(ndcg_by_id)} models in "
            f"{RESCALING.name} - they must describe the same set."
        )
    if len(named) != len(LEGEND):
        raise SystemExit(
            f"{len(named)} non-anonymous markers but {len(LEGEND)} legend entries."
        )

    rows = []

    for marker in named:
        key = (marker["fill"], marker["shape"])
        if key not in LEGEND:
            raise SystemExit(f"Marker {key} is not in the legend map.")
        rows.append({**marker, "model_id": LEGEND[key]})

    taken = {row["model_id"] for row in rows}
    remaining = [
        (model_id, value)
        for model_id, value in ndcg_by_id.items()
        if model_id not in taken
    ]

    # Both sides sorted by score: within the 27 that are left, no two models
    # are closer than the pixel error, so rank order is a safe assignment.
    remaining.sort(key=lambda pair: -pair[1])
    for marker, (model_id, _) in zip(
        sorted(anonymous, key=lambda m: m["y"]), remaining
    ):
        rows.append({**marker, "model_id": model_id})

    for row in rows:
        row["latency_s"] = to_latency(row["x"])
        row["ndcg"] = ndcg_by_id[row["model_id"]]
        row["ndcg_from_pixels"] = to_ndcg(row["y"])
        row["frontier"] = (
            "bi-encoder"
            if row["model_id"] in BI_ENCODER_FRONTIER
            else "cross" if row["model_id"] in CROSS_FRONTIER else ""
        )

    rows.sort(key=lambda r: -r["ndcg"])

    if args.verify:
        worst = 0.0
        for row in rows:
            delta = abs(row["ndcg"] - row["ndcg_from_pixels"])
            worst = max(worst, delta)
            flag = "  <-- legend" if not row["anonymous"] else ""
            print(
                f'{row["ndcg"]:.3f} repo | {row["ndcg_from_pixels"]:.4f} px | '
                f'd={delta:.4f} | {row["latency_s"]:8.3f}s | {row["model_id"]}{flag}'
            )
        print(f"\nLargest disagreement: {worst:.4f} nDCG over {len(rows)} models.")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "latency_s", "ndcg", "frontier", "named"])
        for row in rows:
            writer.writerow(
                [
                    row["model_id"],
                    f'{row["latency_s"]:.4f}',
                    f'{row["ndcg"]:.3f}',
                    row["frontier"],
                    "0" if row["anonymous"] else "1",
                ]
            )

    print(f"Wrote {len(rows)} models to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
