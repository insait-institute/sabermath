# Benchmark vs MTEB Retrieval Correlation

This script compares a hardcoded set of benchmark Overall scores against MTEB Retrieval scores from a CSV file.
It reports:
- Pearson correlation
- Pearson p-value
- Spearman correlation
- Spearman p-value
- Matched model table with benchmark rank and MTEB retrieval rank

## Input CSV

The MTEB CSV file must contain at least these columns:
- Model
- Retrieval

The script also supports the optional column:
- Rank (Borda)

Rank (Borda) is only used as a duplicate tie-breaker. It is not used in the correlations.

## Usage

Run with all benchmark models:
    python script.py --mteb_file path/to/mteb.csv

Run only with the newer presented models:
    python script.py --mteb_file path/to/mteb.csv --new-models-only

## --new-models-only

When this flag is provided, the script excludes these model families from the correlation:
- BGE models
- BERT models
- RoBERTa models
- E5 models
- text-embedding-* models

## Notes
    - Pearson uses the raw benchmark Overall and MTEB Retrieval score values.
    - Spearman uses the same two score columns but ranks them internally.
    - Both scores are higher-is-better, so no sign flip is applied.
    - Models without an MTEB mapping or without a Retrieval score are excluded from the final correlation.
