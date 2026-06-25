from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Final

import pandas as pd
from scipy.stats import pearsonr, spearmanr


MODEL_COL: Final[str] = "Model"
RETRIEVAL_SCORE_COL: Final[str] = "Retrieval"

# Optional: only used for display / duplicate tie-breaking if present.
BORDA_RANK_COL: Final[str] = "Rank (Borda)"

BENCHMARK_OVERALL_RANK_COL: Final[str] = "benchmark_overall_rank"
MTEB_RETRIEVAL_RANK_FULL_COL: Final[str] = "mteb_retrieval_rank_full_csv"
MTEB_RETRIEVAL_RANK_MATCHED_COL: Final[str] = "mteb_retrieval_rank_among_matched"

LINK_RE = re.compile(r"^\[([^\]]+)\]\((.*)\)$")


# Overall column from your LaTeX table.
#
# For rows where `mteb_model_name` is None, the row is not used in the
# MTEB correlation because there is no obvious matching MTEB leaderboard model.
# If you know the exact MTEB model name for a baseline, fill it in.
OUR_BENCHMARK_OVERALL_SCORES: Final[list[dict[str, object]]] = [
    {
        "paper_model_name": "Octen-Embedding-8B",
        "mteb_model_name": "Octen-Embedding-8B",
        "benchmark_overall": 0.636,
    },
    {
        "paper_model_name": "Octen-Embedding-4B",
        "mteb_model_name": "Octen-Embedding-4B",
        "benchmark_overall": 0.632,
    },
    {
        "paper_model_name": "Gemini-Embedding-2",
        "mteb_model_name": "gemini-embedding-2-preview",
        "benchmark_overall": 0.628,
    },
    {
        "paper_model_name": "Qwen3-Embedding-4B",
        "mteb_model_name": "Qwen3-Embedding-4B",
        "benchmark_overall": 0.615,
    },
    {
        "paper_model_name": "Qwen3-Embedding-8B",
        "mteb_model_name": "Qwen3-Embedding-8B",
        "benchmark_overall": 0.611,
    },
    {
        "paper_model_name": "Harrier-OSS-v1-27b",
        "mteb_model_name": "harrier-oss-v1-27b",
        "benchmark_overall": 0.608,
    },
    {
        "paper_model_name": "Gemini-Embedding-001",
        "mteb_model_name": "gemini-embedding-001",
        "benchmark_overall": 0.605,
    },
    {
        "paper_model_name": "KaLM-Embedding-Gemma3-12B-2511",
        "mteb_model_name": "KaLM-Embedding-Gemma3-12B-2511",
        "benchmark_overall": 0.585,
    },
    {
        "paper_model_name": "LLaMa-Embed-Nemotron-8b",
        "mteb_model_name": "llama-embed-nemotron-8b",
        "benchmark_overall": 0.579,
    },
    {
        "paper_model_name": "Qwen3-Embedding-0.6B",
        "mteb_model_name": "Qwen3-Embedding-0.6B",
        "benchmark_overall": 0.575,
    },
    {
        "paper_model_name": "Harrier-OSS-v1-0.6b",
        "mteb_model_name": "harrier-oss-v1-0.6b",
        "benchmark_overall": 0.572,
    },
    {
        "paper_model_name": "Jina-Embeddings-v5-Text-Small",
        "mteb_model_name": "jina-embeddings-v5-text-small",
        "benchmark_overall": 0.570,
    },
    {
        "paper_model_name": "Text-Embedding-3-Large",
        "mteb_model_name": "text-embedding-3-large",
        "benchmark_overall": 0.558,
    },
    {
        "paper_model_name": "Jina-Embeddings-v5-Text-Nano",
        "mteb_model_name": "jina-embeddings-v5-text-nano",
        "benchmark_overall": 0.532,
    },
    {
        "paper_model_name": "EmbeddingGemma-300m",
        "mteb_model_name": "embeddinggemma-300m",
        "benchmark_overall": 0.519,
    },
    {
        "paper_model_name": "Text-Embedding-3-Small",
        "mteb_model_name": "text-embedding-3-small",
        "benchmark_overall": 0.512,
    },
    {
        "paper_model_name": "BGE-m3",
        "mteb_model_name": "bge-m3",
        "benchmark_overall": 0.511,
    },
    {
        "paper_model_name": "Harrier-OSS-v1-270m",
        "mteb_model_name": "harrier-oss-v1-270m",
        "benchmark_overall": 0.498,
    },
    {
        "paper_model_name": "Multilingual-E5-Large",
        "mteb_model_name": "multilingual-e5-large",
        "benchmark_overall": 0.488,
    },
    {
        "paper_model_name": "BERT",
        "mteb_model_name": None,
        "benchmark_overall": 0.357,
    },
    {
        "paper_model_name": "RoBERTa",
        "mteb_model_name": None,
        "benchmark_overall": 0.311,
    },
]


def normalize_mteb_model_name(value: object) -> str:
    """Convert '[name](url)' to 'name'; leave plain names unchanged."""
    if pd.isna(value):
        return ""

    s = str(value).strip()
    match = LINK_RE.match(s)
    if match:
        return match.group(1).strip()
    return s


def canonical_model_name(value: object) -> str:
    """
    Canonical name for matching.

    This is intentionally conservative: it lowercases and normalizes dashes,
    but it does not do fuzzy matching.
    """
    s = normalize_mteb_model_name(value)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[‐‑‒–—−]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def numeric_series(series: pd.Series) -> pd.Series:
    """
    Robust numeric conversion for MTEB CSV columns.

    Handles values like:
      - 54.32
      - "54.32"
      - "54.32%"
      - "-"
      - ""
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "—": pd.NA,
            "N/A": pd.NA,
            "n/a": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
        }
    )
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)

    extracted = cleaned.str.extract(
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        expand=False,
    )

    return pd.to_numeric(extracted, errors="coerce")


def is_excluded_for_new_models_only(row: pd.Series) -> bool:
    """
    Return True for baseline/older model families that should be excluded when
    --new-models-only is used.

    Excluded families:
      - BGE models
      - BERT models
      - RoBERTa models
      - E5 models
      - OpenAI text-embedding models, e.g. text-embedding-3-large/small

    The check uses both the paper display name and the MTEB model name.
    """
    names = [
        canonical_model_name(row.get("paper_model_name")),
        canonical_model_name(row.get("mteb_model_name")),
    ]

    for name in names:
        if not name:
            continue

        # Token-style exclusions. This avoids accidentally excluding unrelated
        # names where these strings appear as part of a larger token.
        if re.search(r"(^|[-_/ ])bge($|[-_/ ])", name):
            return True

        if re.search(r"(^|[-_/ ])bert($|[-_/ ])", name):
            return True

        if re.search(r"(^|[-_/ ])roberta($|[-_/ ])", name):
            return True

        if re.search(r"(^|[-_/ ])e5($|[-_/ ])", name):
            return True

        # OpenAI text embedding models. This intentionally does NOT exclude
        # models such as "jina-embeddings-v5-text-small".
        if name.startswith("text-embedding-"):
            return True

    return False


def build_benchmark_scores(new_models_only: bool = False) -> pd.DataFrame:
    benchmark = pd.DataFrame(OUR_BENCHMARK_OVERALL_SCORES)

    required_cols = {"paper_model_name", "mteb_model_name", "benchmark_overall"}
    missing = required_cols - set(benchmark.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")

    benchmark["benchmark_overall"] = pd.to_numeric(
        benchmark["benchmark_overall"],
        errors="raise",
    )
    benchmark["model_key"] = benchmark["mteb_model_name"].map(canonical_model_name)

    if new_models_only:
        excluded_mask = benchmark.apply(is_excluded_for_new_models_only, axis=1)
        excluded = benchmark.loc[excluded_mask, "paper_model_name"].tolist()

        benchmark = benchmark.loc[~excluded_mask].copy()

        print("Running with --new-models-only.")
        if excluded:
            print("Excluded baseline/OpenAI text models:")
            for name in excluded:
                print(f"  - {name}")
        else:
            print("No models were excluded by --new-models-only.")

        if benchmark.empty:
            raise ValueError(
                "No benchmark models remain after applying --new-models-only."
            )

    mapped = benchmark[benchmark["model_key"] != ""]
    duplicated = mapped[mapped["model_key"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "Duplicate benchmark MTEB mappings:\n"
            + duplicated[
                ["paper_model_name", "mteb_model_name", "model_key"]
            ].to_string(index=False)
        )

    return benchmark


def load_mteb_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MTEB CSV not found: {path}")

    mteb = pd.read_csv(path, encoding="utf-8-sig")

    missing_required = {MODEL_COL, RETRIEVAL_SCORE_COL} - set(mteb.columns)
    if missing_required:
        raise ValueError(
            f"Missing required MTEB columns: {sorted(missing_required)}\n"
            f"Available columns: {list(mteb.columns)}"
        )

    mteb["model_name_from_mteb"] = mteb[MODEL_COL].map(normalize_mteb_model_name)
    mteb["model_key"] = mteb["model_name_from_mteb"].map(canonical_model_name)
    mteb = mteb[mteb["model_key"] != ""].copy()

    mteb[RETRIEVAL_SCORE_COL] = numeric_series(mteb[RETRIEVAL_SCORE_COL])
    mteb["has_retrieval_result"] = mteb[RETRIEVAL_SCORE_COL].notna()

    if BORDA_RANK_COL in mteb.columns:
        mteb[BORDA_RANK_COL] = numeric_series(mteb[BORDA_RANK_COL])

    # If duplicate display names occur, keep the row with best Retrieval score.
    sort_cols = [RETRIEVAL_SCORE_COL]
    ascending = [False]

    # Optional tie-breaker only.
    if BORDA_RANK_COL in mteb.columns:
        sort_cols.append(BORDA_RANK_COL)
        ascending.append(True)

    mteb = (
        mteb.sort_values(
            sort_cols,
            ascending=ascending,
            na_position="last",
        )
        .drop_duplicates(subset=["model_key"], keep="first")
        .copy()
    )

    # Full-MTEB retrieval rank derived from Retrieval score.
    # Higher Retrieval is better, so rank descending.
    mteb[MTEB_RETRIEVAL_RANK_FULL_COL] = mteb[RETRIEVAL_SCORE_COL].rank(
        ascending=False,
        method="average",
    )

    return mteb


def compute_correlations(
    mteb_csv_path: str | Path,
    new_models_only: bool = False,
) -> tuple[float, float, float, float, pd.DataFrame]:
    benchmark_all = build_benchmark_scores(new_models_only=new_models_only)
    mteb = load_mteb_csv(mteb_csv_path)

    unmapped_benchmark_rows = benchmark_all[benchmark_all["model_key"] == ""]
    if not unmapped_benchmark_rows.empty:
        print("Benchmark rows with no MTEB mapping, excluded from correlation:")
        for name in unmapped_benchmark_rows["paper_model_name"].tolist():
            print(f"  - {name}")

    benchmark = benchmark_all[benchmark_all["model_key"] != ""].copy()

    mteb_cols = [
        "model_key",
        "model_name_from_mteb",
        MODEL_COL,
        RETRIEVAL_SCORE_COL,
        MTEB_RETRIEVAL_RANK_FULL_COL,
        "has_retrieval_result",
    ]

    if BORDA_RANK_COL in mteb.columns:
        mteb_cols.append(BORDA_RANK_COL)

    merged = benchmark.merge(
        mteb[mteb_cols],
        on="model_key",
        how="left",
    )

    unmatched = merged.loc[
        merged["model_name_from_mteb"].isna(),
        ["paper_model_name", "mteb_model_name"],
    ]

    if not unmatched.empty:
        print()
        print("Benchmark models not found in the MTEB CSV:")
        for _, row in unmatched.iterrows():
            print(f"  - {row['paper_model_name']} -> {row['mteb_model_name']}")

    matched = merged.loc[merged["model_name_from_mteb"].notna()].copy()
    matched["has_retrieval_result"] = (
        matched["has_retrieval_result"].fillna(False).astype(bool)
    )

    no_retrieval = matched.loc[
        ~matched["has_retrieval_result"],
        ["paper_model_name", "model_name_from_mteb"],
    ]

    if not no_retrieval.empty:
        print()
        print("Matched MTEB rows with no Retrieval score, excluded:")
        for _, row in no_retrieval.iterrows():
            print(f"  - {row['paper_model_name']} -> {row['model_name_from_mteb']}")

    matched = matched.dropna(
        subset=["benchmark_overall", RETRIEVAL_SCORE_COL],
    ).copy()

    if len(matched) < 2:
        raise ValueError(
            "Need at least two matched models with both benchmark Overall "
            "and MTEB Retrieval scores."
        )

    if matched["benchmark_overall"].nunique(dropna=True) < 2:
        raise ValueError(
            "Benchmark Overall scores are constant; correlation is undefined."
        )

    if matched[RETRIEVAL_SCORE_COL].nunique(dropna=True) < 2:
        raise ValueError(
            "MTEB Retrieval scores are constant; correlation is undefined."
        )

    # Ranks are for display/debugging only.
    # Pearson uses the raw scores below.
    matched[BENCHMARK_OVERALL_RANK_COL] = matched["benchmark_overall"].rank(
        ascending=False,
        method="average",
    )
    matched[MTEB_RETRIEVAL_RANK_MATCHED_COL] = matched[RETRIEVAL_SCORE_COL].rank(
        ascending=False,
        method="average",
    )

    # Pearson uses actual score magnitudes.
    pearson_r, pearson_p = pearsonr(
        matched["benchmark_overall"],
        matched[RETRIEVAL_SCORE_COL],
    )

    # Spearman uses the same score columns but internally converts them to ranks.
    # Since both scores are higher-is-better, no sign flip is needed.
    spearman_rho, spearman_p = spearmanr(
        matched["benchmark_overall"],
        matched[RETRIEVAL_SCORE_COL],
    )

    matched = matched.sort_values(
        [BENCHMARK_OVERALL_RANK_COL, "paper_model_name"],
    ).copy()

    return (
        float(pearson_r),
        float(pearson_p),
        float(spearman_rho),
        float(spearman_p),
        matched,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb_file", required=True, type=Path)
    parser.add_argument(
        "--new-models-only",
        action="store_true",
        help=(
            "Only calculate correlations for the newer presented models. "
            "Excludes BGE, BERT, RoBERTa, E5, and OpenAI text-embedding models."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pearson_r, pearson_p, spearman_rho, spearman_p, matched = compute_correlations(
        args.mteb_file,
        new_models_only=args.new_models_only,
    )

    print()
    print("Correlation using benchmark Overall score vs MTEB Retrieval score")
    if args.new_models_only:
        print("Mode: new models only")
    else:
        print("Mode: all benchmark models")
    print(f"Matched models with both scores: {len(matched)}")

    print()
    print("Score-sensitive correlation:")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"Pearson p-value: {pearson_p:.4g}")

    print()
    print("Rank/order correlation using the same score columns:")
    print(f"Spearman rho: {spearman_rho:.4f}")
    print(f"Spearman p-value: {spearman_p:.4g}")

    print()
    print("Notes:")
    print("  - Rank (Borda) is NOT used for either correlation.")
    print("  - Pearson uses benchmark Overall and MTEB Retrieval score magnitudes.")
    print("  - Spearman uses the same two score columns but ranks them internally.")
    print(
        "  - Both Overall and Retrieval are higher-is-better, so no sign flip is used."
    )
    if args.new_models_only:
        print(
            "  - --new-models-only excludes BGE, BERT, RoBERTa, E5, "
            "and OpenAI text-embedding models."
        )

    display_cols = [
        BENCHMARK_OVERALL_RANK_COL,
        "paper_model_name",
        "benchmark_overall",
        RETRIEVAL_SCORE_COL,
        MTEB_RETRIEVAL_RANK_FULL_COL,
    ]

    print()
    print(
        matched[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
