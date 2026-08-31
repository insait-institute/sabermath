# Benchmark vs MTEB Retrieval Correlation

Compares SABER-Math Overall scores against MTEB Retrieval scores from a
leaderboard CSV. It reports:
- Pearson correlation
- Pearson p-value
- Spearman correlation
- Spearman p-value
- Matched model table with benchmark rank and MTEB retrieval rank

## Where the benchmark scores come from

They are **not** stored in this file or anywhere else in the analysis. The
Overall column is read live from `results/evaluation/` through the same code
path that builds the paper's main table
(`sabermath.tables.build_rows`), so the correlation can never be
computed against numbers the tables no longer report. Only the SABER-Math
model key to MTEB leaderboard name mapping is hardcoded, in
`MTEB_MODEL_NAMES`; a `None` there means there is no obvious MTEB counterpart
and the row is left out of the correlation.

Point it at a different set of runs with `--results-dir`.

## Input CSV

The MTEB CSV file must contain at least these columns:
- Model
- Retrieval

The script also supports the optional column:
- Rank (Borda)

Rank (Borda) is only used as a duplicate tie-breaker. It is not used in the correlations.

## Usage

Run with all benchmark models:

```bash
python scripts/analysis/mteb_correlation.py --mteb-file path/to/mteb.csv
```

Run only with the newer presented models:

```bash
python scripts/analysis/mteb_correlation.py --mteb-file path/to/mteb.csv --new-models-only
```

`scripts/report_experiments.py mteb` runs the first form for you, using
`results/mteb/leaderboard.csv` (or `--mteb-file`). The leaderboard export is
not redistributable, so it is not in this repo; the report is skipped with a
note when it is absent.

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
