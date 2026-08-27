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
    # "bm25",
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
    "bm25",
    "google-bert/bert-base-uncased",
    "FacebookAI/roberta-base",
    "microsoft/codebert-base",
    # --- Added 2026-08-23: the rest of the paper's model table, run via
    # load_models.ADDITIONAL_MODELS. Every id below must match that
    # registry's own strings EXACTLY (model_to_file_stem() just does
    # .replace("/", "_") - no other normalization). ---
    # EMBED
    "hanhainebula/reason-embed-qwen3-8b-0928",
    "AQ-MedAI/Diver-Retriever-4B",
    "AQ-MedAI/Diver-Retriever-0.6B",
    "infly/inf-retriever-v1-pro",
    "reasonir/ReasonIR-8B",
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "nvidia/llama-embed-nemotron-8b",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v5-text-small",
    # API (OpenRouter)
    "text-embedding-3-large",
    "text-embedding-3-small",
    # RERANK
    "jhu-clsp/rank1-32b",
    # --- Added 2026-08-26 ---
    "jhu-clsp/rank1-7b",
    "jhu-clsp/rank1-0.5b",
    "inf-x-retriever",
    "qwen3-reranker-8b",
    "qwen3-reranker-4b",
    "qwen3-reranker-0.6b",
    "splade-code-8b",
    "splade-code-0.6b",
    "rader-reranker-7b",
    "diver-grouprank-32b",
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/Reason-ModernColBERT",
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
    "bm25": "BM-25",
    "google-bert/bert-base-uncased": "BERT",
    "FacebookAI/roberta-base": "RoBERTa",
    "microsoft/codebert-base": "CodeBERT",
    # --- Added 2026-08-23 ---
    "hanhainebula/reason-embed-qwen3-8b-0928": "Reason-Embed Qwen3 8B",
    "AQ-MedAI/Diver-Retriever-4B": "Diver-Retriever 4B",
    "AQ-MedAI/Diver-Retriever-0.6B": "Diver-Retriever 0.6B",
    "infly/inf-retriever-v1-pro": "INF-Retriever v1 Pro",
    "reasonir/ReasonIR-8B": "ReasonIR 8B",
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "RaDeR 14B",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "RaDeR 7B",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "RaDeR 3B",
    "nvidia/llama-embed-nemotron-8b": "LLaMA-Embed Nemotron 8B",
    "intfloat/multilingual-e5-large": "Multilingual E5 Large",
    "jinaai/jina-embeddings-v5-text-small": "Jina Small",
    "text-embedding-3-large": "Text-Embedding-3 Large",
    "text-embedding-3-small": "Text-Embedding-3 Small",
    "jhu-clsp/rank1-32b": "Rank1 32B",
    "jhu-clsp/rank1-7b": "Rank1 7B",
    "jhu-clsp/rank1-0.5b": "Rank1 0.5B",
    "inf-x-retriever": "INF-X-Retriever",
    "qwen3-reranker-8b": "Qwen3-Reranker 8B",
    "qwen3-reranker-4b": "Qwen3-Reranker 4B",
    "qwen3-reranker-0.6b": "Qwen3-Reranker 0.6B",
    "splade-code-8b": "SPLADE-Code 8B",
    "splade-code-0.6b": "SPLADE-Code 0.6B",
    "rader-reranker-7b": "RaDeR-Reranker 7B",
    "diver-grouprank-32b": "Diver-GroupRank 32B",
    "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT",
    "lightonai/Reason-ModernColBERT": "Reason-ModernColBERT",
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
    "bm25": "*",
    # --- Added 2026-08-23 ---
    # RaDeR family (Raderspace bi-encoders + its LoRA reranker)
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "d",
    "rader-reranker-7b": "d",
    # Diver family (AQ-MedAI retrievers + the groupwise reranker)
    "AQ-MedAI/Diver-Retriever-4B": "<",
    "AQ-MedAI/Diver-Retriever-0.6B": "<",
    "diver-grouprank-32b": "<",
    # Qwen-architecture derivatives - share Qwen's diamond marker
    "hanhainebula/reason-embed-qwen3-8b-0928": "D",
    "qwen3-reranker-8b": "D",
    "qwen3-reranker-4b": "D",
    "qwen3-reranker-0.6b": "D",
    # SPLADE (naver)
    "splade-code-8b": "P",
    "splade-code-0.6b": "P",
    # ColBERT (lightonai)
    "lightonai/GTE-ModernColBERT-v1": "h",
    "lightonai/Reason-ModernColBERT": "h",
    # Standalone entities. matplotlib's filled-shape alphabet
    # (o v ^ < > 8 s p * h H D d P X) is exhausted by the families above -
    # these reuse a shape from an unrelated family, relying on color alone
    # to distinguish (same convention this file already uses within a
    # family). Do NOT use '1'/'2'/'3'/'4'/'+'/'x'/'|'/'_' - those are
    # unfilled line-only markers in matplotlib, and every scatter call
    # here sets edgecolors="none", linewidths=0 (deliberate, for the
    # filled shapes) - on a line-only marker that draws NOTHING (confirmed
    # visually: they rendered as blank/invisible in
    # plots/maths_vs_words_all_models.pdf).
    "jhu-clsp/rank1-32b": ">",
    "jhu-clsp/rank1-7b": ">",  # shares rank1-32b's triangle_right
    "jhu-clsp/rank1-0.5b": ">",  # shares rank1-32b's triangle_right
    "inf-x-retriever": "8",  # shares inf-retriever-v1-pro's octagon
    "infly/inf-retriever-v1-pro": "8",  # shares BAAI/bge-m3's octagon
    "nvidia/llama-embed-nemotron-8b": "v",  # shares RoBERTa's triangle_down
    "intfloat/multilingual-e5-large": "^",  # shares the Google family's triangle_up
    "reasonir/ReasonIR-8B": "o",  # shares approach0's circle
    "jinaai/jina-embeddings-v5-text-small": "p",
    # OpenAI (via OpenRouter) - shares Octen's square
    "text-embedding-3-large": "s",
    "text-embedding-3-small": "s",
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
    "bm25": "#757575",
    # --- Added 2026-08-23 ---
    # RaDeR family: deep red/maroon, darkest = largest model
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "#b71c1c",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "#e53935",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "#ff8a80",
    "rader-reranker-7b": "#880e4f",
    # Diver family: blue
    "AQ-MedAI/Diver-Retriever-4B": "#1565c0",
    "AQ-MedAI/Diver-Retriever-0.6B": "#42a5f5",
    "diver-grouprank-32b": "#0d47a1",
    # Qwen-architecture derivatives: purple/magenta family (shares the hue
    # with Qwen3-Embedding, distinguished by shade)
    "hanhainebula/reason-embed-qwen3-8b-0928": "#d500f9",
    "qwen3-reranker-8b": "#4a0080",
    "qwen3-reranker-4b": "#7b1fa2",
    "qwen3-reranker-0.6b": "#ce93d8",
    # SPLADE: teal
    "splade-code-8b": "#00838f",
    "splade-code-0.6b": "#26c6da",
    # ColBERT: brown/amber
    "lightonai/GTE-ModernColBERT-v1": "#6d4c41",
    "lightonai/Reason-ModernColBERT": "#a1887f",
    # Standalone entities
    "jhu-clsp/rank1-32b": "#c62828",
    "jhu-clsp/rank1-7b": "#e53935",
    "jhu-clsp/rank1-0.5b": "#ef9a9a",
    "inf-x-retriever": "#26a69a",
    "infly/inf-retriever-v1-pro": "#00695c",
    "nvidia/llama-embed-nemotron-8b": "#558b2f",
    "intfloat/multilingual-e5-large": "#546e7a",
    "reasonir/ReasonIR-8B": "#f57f17",
    "jinaai/jina-embeddings-v5-text-small": "#f48fb1",
    # OpenAI (via OpenRouter): indigo
    "text-embedding-3-large": "#1a237e",
    "text-embedding-3-small": "#3949ab",
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

args = parser.parse_args()
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
    baseline_path = Path("similarities") / f"{model_name}.json"
    path = baseline_path

    if INSTRUCTION_SUFFIX is not None:
        instructed_path = Path("similarities") / f"{model_name}__{INSTRUCTION_SUFFIX}.json"
        if instructed_path.exists():
            path = instructed_path
        else:
            # jaccard/tf-idf/approach0 are excluded from instructions
            # entirely (see load_models.py) and never get a __p1/p2/p3
            # file - fall back to the plain baseline for just this model
            # rather than erroring the whole plot over 3 known exceptions.
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


# Models whose scoring is coarse/discrete enough that an exact
# math_score == text_score tie is expected, not a sign of a bug - skipped
# rather than raised. jaccard was the original case (small-integer set-ratio
# scores). Two more confirmed by direct inspection of similarities/*.json
# after the 2026-08-23 model sweep:
#   - diver-grouprank-32b: GroupRankProcessor's discrete rank-fraction
#     scoring (group_size=20) - 23/969 targets tied.
#   - infly/inf-retriever-v1-pro: loads in fp16 (see load_models.py's
#     _build_inf_retriever_processor) with an already narrow, anisotropic
#     similarity band - fp16's quantization step near 1.0 collapses
#     genuinely-different scores onto the same representable float -
#     9/969 targets tied.
# A fourth confirmed the same way once inf-x-retriever was added to this
# file's model lists (that addition didn't also add it here, which crashed
# --all outright - not related to --instruction, reproduces with neither
# flag set):
#   - inf-x-retriever: shares inf-retriever-v1-pro's byte-identical
#     retriever backend (see scripts/run_rerankers.py's own comment on
#     INFXRetrieverProcessor) - same fp16/anisotropy tie behavior -
#     6/969 targets tied.
# Any OTHER model hitting this is still treated as a real bug (raised
# below) - an unexpected tie in a continuous-score model is worth
# investigating, not silently skipping.
MODELS_WITH_EXPECTED_TIES = {
    "jaccard",
    "diver-grouprank-32b",
    "infly/inf-retriever-v1-pro",
    "inf-x-retriever",
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

    if legend_outside:
        # The in-plot "lower left" legend (below) works for a handful of
        # models, but with all ~40+ it needs so many rows that it swallows
        # entire domain columns (confirmed visually: it covered all of
        # Algebra and part of Calculus in
        # plots/maths_vs_words_all_models.pdf). Below the axes, spread
        # wide and shallow instead - keeps every data point visible and
        # scales to any model count by just adding columns.
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
if INSTRUCTION_SUFFIX is not None:
    # Never silently overwrite the baseline plot - same reasoning as
    # similarities/<model>__p1.json not overwriting similarities/<model>.json.
    output_filename = output_filename.replace(".pdf", f"__{INSTRUCTION_SUFFIX}.pdf")

fig, ax = plot_maths_greater_than_words_points(
    percentages_by_model,
    plot_model_ids=PLOT_MODEL_IDS,
    display_names=display_names,
    marker_symbols=marker_symbols,
    model_colors=model_colors,
    y_min=0.0,
    legend_fontsize=12 if args.all else 20,
    # --all: wide+shallow (many columns) below the axes, via
    # legend_outside - a tall narrow in-plot legend at this model count
    # would cover real data columns. Selected-models keeps the original
    # single-column in-plot legend.
    legend_ncol=6 if args.all else 1,
    legend_outside=args.all,
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
