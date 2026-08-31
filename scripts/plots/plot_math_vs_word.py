#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns
import yaml

from sabermath.figures import (
    ALL_MODEL_IDS,
    DEFAULT_MODEL_COLORS,
    DEFAULT_MODEL_DISPLAY_NAMES,
    DEFAULT_MODEL_MARKER_SYMBOLS,
    MATH_TOKEN_RATIO_MODEL_ID,
)
from sabermath.math_vs_word.sim_helpers import get_math_words_tokens
from sabermath.math_vs_word import PLOTS_DIR, SIMILARITIES_DIR

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

MODEL_IDS = [
    "reason-rewriter-reason-embed-8b",  # ReasonRewriter + ReasonEmbed-Qwen3-8B
    "hanhainebula/reason-embed-qwen3-8b-0928",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes", 
    "AQ-MedAI/Diver-Retriever-4B", 
    "google/gemini-embedding-2",
    "Qwen/Qwen3-Embedding-8B",
    "google/embeddinggemma-300m", 
    "BAAI/bge-m3",
    "approach0",
    "tf-idf",
    "google-bert/bert-base-uncased",  
]

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all models in ALL_MODEL_IDS instead of the smaller default MODEL_IDS list.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Plot the instructed similarity files instead of the plain "
        "baseline ones - e.g. --instruction 1 (or p1) reads "
        "similarities/<model>__p1.json instead of similarities/<model>.json. "
        "Accepts 1/2/3 or p1/p2/p3. jaccard/tf-idf/approach0 have no "
        "instructed variant (see load_models.py's instruction-exclusion "
        "reasons) - those fall back to their plain baseline file with a "
        "printed notice rather than erroring the whole plot.",
    )

    args = parser.parse_args(argv)
    config_file = args.config_file

    def _normalize_instruction_suffix(value: str | None) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        if key.startswith("p"):
            key = key[1:]
        if key not in {"1", "2", "3"}:
            raise ValueError(
                f"--instruction must be 1/2/3 or p1/p2/p3, got {value!r}"
            )
        return f"p{key}"

    INSTRUCTION_SUFFIX = _normalize_instruction_suffix(args.instruction)

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
        baseline_path = SIMILARITIES_DIR / f"{model_name}.json"
        path = baseline_path

        if INSTRUCTION_SUFFIX is not None:
            instructed_path = SIMILARITIES_DIR / f"{model_name}__{INSTRUCTION_SUFFIX}.json"
            if instructed_path.exists():
                path = instructed_path
            else:
                print(
                    f"[~] No {instructed_path.name} for {model_id!r} - "
                    f"falling back to its baseline file ({baseline_path.name})."
                )

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

    MODELS_WITH_EXPECTED_TIES = {
        "jaccard",
        "diver-grouprank-32b",
        "infly/inf-retriever-v1-pro",
        "inf-x-retriever",
        "retro-star-32b",
        "retro-star-32b-rewritten",
    }

    def aggregate_maths_greater_than_words_by_domain(
        content: Mapping[str, Mapping[str, Any]],
        *,
        id_to_domain: Mapping[str, str],
        model_id: str,
    ):
        m_greater_w_counts = {group: 0 for group in GROUP_NAMES}
        totals = {group: 0 for group in GROUP_NAMES}

        skipped_ties = 0

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
                if model_id in MODELS_WITH_EXPECTED_TIES:
                    skipped_ties += 1
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

        if skipped_ties:
            print(f"Skipped {skipped_ties} tied text/math examples for {model_id!r}.")

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
        legend_outside: bool = False,
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

        small = n_scatter_models <= 12 and not legend_outside
        fig_width = 15 if small else 18
        fig_height = 8 if small else 9
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

        ax.set_facecolor((0.97, 0.97, 0.97))

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

        if legend_outside:
            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                bbox_transform=ax.transAxes,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                borderaxespad=0.0,
            )
        else:
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

        if not legend_outside:
            plt.tight_layout()

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    if INSTRUCTION_SUFFIX is not None:
        # Never silently overwrite the baseline plot.
        output_filename = output_filename.replace(".pdf", f"__{INSTRUCTION_SUFFIX}.pdf")

    fig, ax = plot_maths_greater_than_words_points(
        percentages_by_model,
        plot_model_ids=PLOT_MODEL_IDS,
        display_names=display_names,
        marker_symbols=marker_symbols,
        model_colors=model_colors,
        y_min=0.0,
        legend_fontsize=12 if args.all else 17,
        legend_ncol=6 if args.all else 1,
        legend_outside=args.all,
        output_path=str(PLOTS_DIR / output_filename),
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

if __name__ == "__main__":
    main()
