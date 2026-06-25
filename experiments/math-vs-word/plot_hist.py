import argparse
import json
from pathlib import Path
from datasets import load_dataset
from typing import Any, Mapping

import numpy as np
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sim_helpers import get_math_words_tokens

# ---------------------------------------------------------------------
# Default smaller list of models to include in the plot.
# This is used unless you pass --all.
# ---------------------------------------------------------------------

MODEL_IDS = [
    "approach0",
    "Octen/Octen-Embedding-8B",
    # "Octen/Octen-Embedding-4B",
    "google/gemini-embedding-2",
    # "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    # "microsoft/harrier-oss-v1-27b",
    # "google/gemini-embedding-001",
    # "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Qwen/Qwen3-Embedding-0.6B",
    # "microsoft/harrier-oss-v1-0.6b",
    # "jinaai/jina-embeddings-v5-text-nano",
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
    # "microsoft/harrier-oss-v1-270m",
    "tf-idf",
    # "jaccard",
    "google-bert/bert-base-uncased",
    # "FacebookAI/roberta-base",
    # "microsoft/codebert-base",
]

# ---------------------------------------------------------------------
# Full list of models to include when --all is passed.
# Make sure each model here has a corresponding JSON file in similarities/.
# ---------------------------------------------------------------------

ALL_MODEL_IDS = [
    "approach0",
    "Octen/Octen-Embedding-8B",
    "Octen/Octen-Embedding-4B",
    "google/gemini-embedding-2",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "microsoft/harrier-oss-v1-27b",
    "google/gemini-embedding-001",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Qwen/Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-0.6b",
    "jinaai/jina-embeddings-v5-text-nano",
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
    "microsoft/harrier-oss-v1-270m",
    "tf-idf",
    "jaccard",
    "google-bert/bert-base-uncased",
    "FacebookAI/roberta-base",
    "microsoft/codebert-base",
]

MATH_TOKEN_RATIO_MODEL_ID = "target_math_token_ratio"

# ---------------------------------------------------------------------
# Display names for all known models.
# These can be overridden from the YAML config with `model_display_names`.
# ---------------------------------------------------------------------

DEFAULT_MODEL_DISPLAY_NAMES = {
    "approach0": "Approach Zero",
    "Octen/Octen-Embedding-8B": "Octen 8B",
    "Octen/Octen-Embedding-4B": "Octen 4B",
    "google/gemini-embedding-2": "Gemini 2",
    "Qwen/Qwen3-Embedding-4B": "Qwen3 4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen3 8B",
    "microsoft/harrier-oss-v1-27b": "Harrier 27B",
    "google/gemini-embedding-001": "Gemini 001",
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "KaLM Gemma3 12B",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3 0.6B",
    "microsoft/harrier-oss-v1-0.6b": "Harrier 0.6B",
    "jinaai/jina-embeddings-v5-text-nano": "Jina Nano",
    "google/embeddinggemma-300m": "Gemma 300M",
    "BAAI/bge-m3": "BGE-M3",
    "microsoft/harrier-oss-v1-270m": "Harrier 270M",
    "tf-idf": "TF-IDF",
    "jaccard": "Jaccard",
    "google-bert/bert-base-uncased": "BERT",
    "FacebookAI/roberta-base": "RoBERTa",
    "microsoft/codebert-base": "CodeBERT",
    MATH_TOKEN_RATIO_MODEL_ID: "Math-token ratio",
}

# ---------------------------------------------------------------------
# Markers for all models.
# Models from the same family/source intentionally share a marker.
# Individual models are distinguished by color.
#
# The math-token ratio is NOT plotted as a marker; it is plotted as a
# dashed horizontal line within each domain column.
# ---------------------------------------------------------------------

DEFAULT_MODEL_MARKER_SYMBOLS = {
    # Custom / proposed approach
    "approach0": "o",
    # Octen models
    "Octen/Octen-Embedding-8B": "s",
    "Octen/Octen-Embedding-4B": "s",
    # Google models
    "google/gemini-embedding-2": "^",
    "google/gemini-embedding-001": "^",
    "google/embeddinggemma-300m": "^",
    "google-bert/bert-base-uncased": "^",
    # Qwen models
    "Qwen/Qwen3-Embedding-8B": "D",
    "Qwen/Qwen3-Embedding-4B": "D",
    "Qwen/Qwen3-Embedding-0.6B": "D",
    # Microsoft models
    "microsoft/harrier-oss-v1-27b": "X",
    "microsoft/harrier-oss-v1-0.6b": "X",
    "microsoft/harrier-oss-v1-270m": "X",
    "microsoft/codebert-base": "X",
    # Tencent models
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "H",
    # Jina models
    "jinaai/jina-embeddings-v5-text-nano": "p",
    # BAAI models
    "BAAI/bge-m3": "8",
    # Meta/Facebook models
    "FacebookAI/roberta-base": "v",
    # Lexical/string baselines
    "tf-idf": "*",
    "jaccard": "*",
}

# ---------------------------------------------------------------------
# Colors for all known models.
# These can be overridden from the YAML config with `model_colors`.
#
# Models sharing the same marker get colors from related but visibly
# distinguishable color families.
# ---------------------------------------------------------------------

DEFAULT_MODEL_COLORS = {
    # Custom / proposed approach
    "approach0": "#4B4848",
    # Octen models: vivid blue/cyan family
    "Octen/Octen-Embedding-8B": "#0066ff",
    "Octen/Octen-Embedding-4B": "#00b4d8",
    # Google models: bright green/lime/teal family
    "google/gemini-embedding-2": "#00a651",
    "google/gemini-embedding-001": "#7bd000",
    "google/embeddinggemma-300m": "#00c2a8",
    "google-bert/bert-base-uncased": "#38ef7d",
    # Qwen models: saturated purple/magenta family
    "Qwen/Qwen3-Embedding-8B": "#6a00ff",
    "Qwen/Qwen3-Embedding-4B": "#b100ff",
    "Qwen/Qwen3-Embedding-0.6B": "#ff4fd8",
    # Microsoft models: red/orange/yellow family
    "microsoft/harrier-oss-v1-27b": "#ff1744",
    "microsoft/harrier-oss-v1-0.6b": "#ff6d00",
    "microsoft/harrier-oss-v1-270m": "#ffb300",
    "microsoft/codebert-base": "#c51162",
    # Tencent
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "#00acc1",
    # Jina
    "jinaai/jina-embeddings-v5-text-nano": "#ec407a",
    # BAAI
    "BAAI/bge-m3": "#8d6e63",
    # Meta/Facebook
    "FacebookAI/roberta-base": "#1877f2",
    # Lexical/string baselines: distinguishable neutral colors
    "tf-idf": "#5f6368",
    "jaccard": "#9e9e9e",
    # Reference statistic: dashed line color
    MATH_TOKEN_RATIO_MODEL_ID: "#263238",
}

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
parser.add_argument(
    "--all",
    action="store_true",
    help="Plot all models in ALL_MODEL_IDS instead of the smaller default MODEL_IDS list.",
)

args = parser.parse_args()
config_file = args.config_file

SELECTED_MODEL_IDS = ALL_MODEL_IDS if args.all else MODEL_IDS
PLOT_MODEL_IDS = SELECTED_MODEL_IDS + [MATH_TOKEN_RATIO_MODEL_ID]

DOMAIN_ORDER = [
    "Algebra",
    "Calculus and Analysis",
    "Combinatorics",
    "Geometry",
    "Number Theory",
]

GROUP_NAMES = DOMAIN_ORDER + ["All"]

TEXT_KEY = "pr_text_vs_candidates"
MATH_KEY = "pr_math_vs_candidates"


def model_to_file_stem(model_id: str) -> str:
    return model_id.replace("/", "_")


def model_to_display_name(
    model_id: str,
    display_names: Mapping[str, str] | None = None,
) -> str:
    if display_names is not None and model_id in display_names:
        return display_names[model_id]

    if model_id in DEFAULT_MODEL_DISPLAY_NAMES:
        return DEFAULT_MODEL_DISPLAY_NAMES[model_id]

    if model_id == MATH_TOKEN_RATIO_MODEL_ID:
        return "Math-token ratio"

    return model_id.split("/")[-1]


def model_to_marker_symbol(
    model_id: str,
    marker_symbols: Mapping[str, str] | None = None,
) -> str:
    if marker_symbols is not None and model_id in marker_symbols:
        return marker_symbols[model_id]

    if model_id in DEFAULT_MODEL_MARKER_SYMBOLS:
        return DEFAULT_MODEL_MARKER_SYMBOLS[model_id]

    raise KeyError(
        f"No marker symbol was provided for model {model_id!r}. "
        "Add it to DEFAULT_MODEL_MARKER_SYMBOLS or pass it via "
        "`model_marker_symbols` in the YAML config."
    )


def model_to_color(
    model_id: str,
    model_colors: Mapping[str, str] | None = None,
) -> str:
    if model_colors is not None and model_id in model_colors:
        return model_colors[model_id]

    if model_id in DEFAULT_MODEL_COLORS:
        return DEFAULT_MODEL_COLORS[model_id]

    raise KeyError(
        f"No color was provided for model {model_id!r}. "
        "Add it to DEFAULT_MODEL_COLORS or pass it via "
        "`model_colors` in the YAML config."
    )


def load_similarity_content(model_id: str) -> dict[str, dict[str, Any]]:
    model_name = model_to_file_stem(model_id)
    path = Path("similarities") / f"{model_name}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find similarity file for model {model_id!r}: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_first_domain(domains):
    if isinstance(domains, (list, tuple)):
        if len(domains) == 0:
            raise ValueError("Encountered empty domains list.")
        return str(domains[0]).strip()

    if domains is None:
        raise ValueError("Encountered None domain.")

    return str(domains).strip()


def build_id_to_domain(
    *,
    all_content_ids: set[str],
    targets_dataset_name: str,
) -> dict[str, str]:
    targets = load_dataset(targets_dataset_name, split="train").select_columns(
        ["id", "domains"]
    )

    id_to_domain = {}

    for target_id, domains in zip(targets["id"], targets["domains"]):
        target_id = str(target_id)

        if target_id in all_content_ids:
            id_to_domain[target_id] = get_first_domain(domains)

    missing_ids = [
        target_id for target_id in all_content_ids if target_id not in id_to_domain
    ]

    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} ids from the similarity files were not found "
            f"in the targets dataset. Examples: {missing_ids[:10]}"
        )

    return id_to_domain


def target_math_token_ratio(target: Mapping[str, Any]) -> float:
    """
    Given one target from the targets dataset, return:

        number_of_mathematics_tokens / total_number_of_tokens

    The returned value must be a float in [0, 1].
    """

    target_math_tokens, target_text_tokens = get_math_words_tokens(
        target["problem_math_expr"],
        target["problem_text_only"],
    )

    total_tokens = len(target_math_tokens) + len(target_text_tokens)

    if total_tokens == 0:
        raise ValueError(
            f"Target {target.get('id', '<unknown>')!r} has zero total tokens."
        )

    return float(len(target_math_tokens) / total_tokens)


def aggregate_maths_greater_than_words_by_domain(
    content: Mapping[str, Mapping[str, Any]],
    *,
    id_to_domain: Mapping[str, str],
    model_id: str,
):
    m_greater_w_counts = {group: 0 for group in GROUP_NAMES}
    totals = {group: 0 for group in GROUP_NAMES}

    skipped_jaccard_ties = 0

    for target_id, row in content.items():
        target_id = str(target_id)

        missing_keys = [key for key in [TEXT_KEY, MATH_KEY] if key not in row]

        if missing_keys:
            raise KeyError(
                f"Model {model_id!r}, target {target_id!r} is missing keys: "
                f"{missing_keys}"
            )

        if target_id not in id_to_domain:
            raise KeyError(f"Model {model_id!r}, target {target_id!r} has no domain.")

        domain = str(id_to_domain[target_id]).strip()

        if domain not in DOMAIN_ORDER:
            raise ValueError(
                f"Model {model_id!r}, target {target_id!r} has unknown domain "
                f"{domain!r}. Expected one of: {DOMAIN_ORDER}"
            )

        text_score = float(row[TEXT_KEY])
        math_score = float(row[MATH_KEY])

        if not np.isfinite(text_score) or not np.isfinite(math_score):
            raise ValueError(
                f"Model {model_id!r}, target {target_id!r} has non-finite scores: "
                f"text={text_score}, math={math_score}"
            )

        if text_score == math_score:
            if model_id == "jaccard":
                skipped_jaccard_ties += 1
                continue

            raise ValueError(
                f"Model {model_id!r}, target {target_id!r} has tied text/math "
                f"scores: text={text_score}, math={math_score}"
            )

        is_math_greater = math_score > text_score

        totals["All"] += 1
        totals[domain] += 1

        if is_math_greater:
            m_greater_w_counts["All"] += 1
            m_greater_w_counts[domain] += 1

    if skipped_jaccard_ties:
        print(f"Skipped {skipped_jaccard_ties} tied text/math examples for jaccard.")

    percentages = {
        group: (
            100.0 * m_greater_w_counts[group] / totals[group]
            if totals[group] > 0
            else np.nan
        )
        for group in GROUP_NAMES
    }

    return percentages, m_greater_w_counts, totals


def aggregate_target_math_token_ratio_by_domain(
    *,
    all_content_ids: set[str],
    targets_dataset_name: str,
):
    targets = load_dataset(targets_dataset_name, split="train")

    ratio_sums = {group: 0.0 for group in GROUP_NAMES}
    totals = {group: 0 for group in GROUP_NAMES}

    seen_ids = set()

    for target in targets:
        target_id = str(target["id"])

        if target_id not in all_content_ids:
            continue

        seen_ids.add(target_id)

        domain = get_first_domain(target["domains"])

        if domain not in DOMAIN_ORDER:
            raise ValueError(
                f"Target {target_id!r} has unknown domain {domain!r}. "
                f"Expected one of: {DOMAIN_ORDER}"
            )

        ratio = target_math_token_ratio(target)

        if ratio is None:
            raise ValueError(
                "target_math_token_ratio returned None. "
                "Fill in target_math_token_ratio(target) before running."
            )

        ratio = float(ratio)

        if not np.isfinite(ratio):
            raise ValueError(
                f"Target {target_id!r} has non-finite math-token ratio: {ratio}"
            )

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"Target {target_id!r} has invalid math-token ratio {ratio}. "
                "Expected a value in [0, 1]."
            )

        ratio_sums["All"] += ratio
        ratio_sums[domain] += ratio

        totals["All"] += 1
        totals[domain] += 1

    missing_ids = [
        target_id for target_id in all_content_ids if target_id not in seen_ids
    ]

    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} ids from the similarity files were not found "
            f"in the targets dataset while computing math-token ratios. "
            f"Examples: {missing_ids[:10]}"
        )

    average_ratios = {
        group: (ratio_sums[group] / totals[group] if totals[group] > 0 else np.nan)
        for group in GROUP_NAMES
    }

    average_ratio_percentages = {
        group: (
            100.0 * average_ratios[group]
            if np.isfinite(average_ratios[group])
            else np.nan
        )
        for group in GROUP_NAMES
    }

    return average_ratio_percentages, average_ratios, totals


def plot_maths_greater_than_words_points(
    percentages_by_model: Mapping[str, Mapping[str, float]],
    *,
    plot_model_ids: list[str],
    display_names: Mapping[str, str] | None = None,
    marker_symbols: Mapping[str, str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    y_min: float = 0.0,
    y_max: float = 102.0,
    legend_fontsize: float = 17.0,
    legend_ncol: int = 1,
    output_path: str = "plots/maths_greater_than_words_all_models_with_math_token_ratio.pdf",
):
    sns.set_theme(style="white")

    group_display_labels = {
        "All": "All",
        "Algebra": "Algebra",
        "Calculus and Analysis": "Calculus\nand Analysis",
        "Combinatorics": "Combinatorics",
        "Geometry": "Geometry",
        "Number Theory": "Number\nTheory",
    }

    group_spacing = 1.45
    x = np.arange(len(GROUP_NAMES)) * group_spacing

    # The math-token ratio is drawn as a dashed line, not as a scatter marker.
    scatter_model_ids = [
        model_id for model_id in plot_model_ids if model_id != MATH_TOKEN_RATIO_MODEL_ID
    ]

    n_scatter_models = len(scatter_model_ids)
    cluster_width = 0.55 if n_scatter_models <= 12 else 0.85

    if n_scatter_models == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -cluster_width / 2,
            cluster_width / 2,
            n_scatter_models,
        )

    fig_width = 15 if n_scatter_models <= 12 else 18
    fig_height = 8 if n_scatter_models <= 12 else 9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    ax.set_facecolor((0.97, 0.97, 0.97))

    # -----------------------------------------------------------------
    # Plot regular models as scatter markers.
    # No black borders are used.
    # -----------------------------------------------------------------

    for i, model_id in enumerate(scatter_model_ids):
        vals = [percentages_by_model[model_id][group] for group in GROUP_NAMES]

        marker_symbol = model_to_marker_symbol(
            model_id,
            marker_symbols=marker_symbols,
        )

        model_color = model_to_color(
            model_id,
            model_colors=model_colors,
        )

        ax.scatter(
            x + offsets[i],
            vals,
            label=model_to_display_name(model_id, display_names=display_names),
            color=model_color,
            marker=marker_symbol,
            s=120,
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )

    # -----------------------------------------------------------------
    # Plot math-token ratio as dashed horizontal line segments.
    # Each segment spans the corresponding domain column.
    # -----------------------------------------------------------------

    if MATH_TOKEN_RATIO_MODEL_ID in plot_model_ids:
        math_ratio_vals = [
            percentages_by_model[MATH_TOKEN_RATIO_MODEL_ID][group]
            for group in GROUP_NAMES
        ]

        math_ratio_color = model_to_color(
            MATH_TOKEN_RATIO_MODEL_ID,
            model_colors=model_colors,
        )

        math_ratio_label = model_to_display_name(
            MATH_TOKEN_RATIO_MODEL_ID,
            display_names=display_names,
        )

        # Width of the dashed segment within each domain column.
        # This is intentionally wider than the marker cluster.
        ratio_line_half_width = group_spacing * 0.34

        for j, y_val in enumerate(math_ratio_vals):
            ax.hlines(
                y=y_val,
                xmin=x[j] - ratio_line_half_width,
                xmax=x[j] + ratio_line_half_width,
                colors=math_ratio_color,
                linestyles=(0, (5, 3)),
                linewidth=2.6,
                label=math_ratio_label if j == 0 else "_nolegend_",
                zorder=2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [group_display_labels[group] for group in GROUP_NAMES],
        rotation=0,
        ha="center",
        fontsize=22,
        linespacing=1.15,
    )

    ax.tick_params(axis="y", labelsize=23)

    ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks(np.arange(0, 101, 10))

    ax.set_xlim(
        x[0] - group_spacing * 0.55,
        x[-1] + group_spacing * 0.55,
    )

    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    sns.despine(ax=ax, left=True, bottom=True)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.015, 0.015),
        bbox_transform=ax.transAxes,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
        borderaxespad=0.0,
    )
    legend.set_zorder(10)

    plt.tight_layout()

    Path("plots").mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return fig, ax


with open(config_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if config is None:
    config = {}

targets_maths_words_dataset = config["hf_datasets"]["targets_maths_words_fixed"]

contents_by_model = {
    model_id: load_similarity_content(model_id) for model_id in SELECTED_MODEL_IDS
}

all_content_ids = set()
for content in contents_by_model.values():
    all_content_ids.update(str(target_id) for target_id in content.keys())

id_to_domain = build_id_to_domain(
    all_content_ids=all_content_ids,
    targets_dataset_name=targets_maths_words_dataset,
)

percentages_by_model = {}
counts_by_model = {}
totals_by_model = {}

for model_id, content in contents_by_model.items():
    percentages, counts, totals = aggregate_maths_greater_than_words_by_domain(
        content,
        id_to_domain=id_to_domain,
        model_id=model_id,
    )

    percentages_by_model[model_id] = percentages
    counts_by_model[model_id] = counts
    totals_by_model[model_id] = totals


math_token_ratio_percentages, math_token_ratio_averages, math_token_ratio_totals = (
    aggregate_target_math_token_ratio_by_domain(
        all_content_ids=all_content_ids,
        targets_dataset_name=targets_maths_words_dataset,
    )
)

percentages_by_model[MATH_TOKEN_RATIO_MODEL_ID] = math_token_ratio_percentages
totals_by_model[MATH_TOKEN_RATIO_MODEL_ID] = math_token_ratio_totals

display_names = {
    **DEFAULT_MODEL_DISPLAY_NAMES,
    **config.get("model_display_names", {}),
}

marker_symbols = {
    **DEFAULT_MODEL_MARKER_SYMBOLS,
    **config.get("model_marker_symbols", {}),
}

model_colors = {
    **DEFAULT_MODEL_COLORS,
    **config.get("model_colors", {}),
}

output_filename = (
    "maths_vs_words_all_models.pdf"
    if args.all
    else "maths_vs_words_selected_models.pdf"
)

fig, ax = plot_maths_greater_than_words_points(
    percentages_by_model,
    plot_model_ids=PLOT_MODEL_IDS,
    display_names=display_names,
    marker_symbols=marker_symbols,
    model_colors=model_colors,
    y_min=0.0,
    legend_fontsize=15 if args.all else 20,
    legend_ncol=2 if args.all else 1,
    output_path=str(Path("plots") / output_filename),
)

print("\nPercentage of targets with M>W:")
for model_id in SELECTED_MODEL_IDS:
    print(f"\n{model_id}")
    for group in GROUP_NAMES:
        pct = percentages_by_model[model_id][group]
        count = counts_by_model[model_id][group]
        total = totals_by_model[model_id][group]
        print(f"  {group}: {pct:.1f}% ({count}/{total})")

print("\nAverage target math-token ratio:")
for group in GROUP_NAMES:
    avg_ratio = math_token_ratio_averages[group]
    avg_ratio_pct = math_token_ratio_percentages[group]
    total = math_token_ratio_totals[group]
    print(f"  {group}: {avg_ratio:.4f} = {avg_ratio_pct:.1f}% over {total} targets")
