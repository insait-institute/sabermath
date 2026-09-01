# SABER-Math vs MTEB Retrieval

Correlates each model's SABER-Math Overall score against its MTEB Retrieval score.

## Prerequisites

- `python -m pip install -e .` (needs `pandas` and `scipy`).
- Runs in `results/evaluation/`. Point elsewhere with `--results-dir`.
- The MTEB leaderboard export, checked in at `results/mteb/leaderboard.csv`. Pass `--mteb-file` to correlate against a different one. Required columns:

  | Column | Use |
  |---|---|
  | `Model` | joined to `MTEB_MODEL_NAMES` in `scripts/analysis/mteb_correlation.py` |
  | `Retrieval` | the MTEB score |
  | `Rank (Borda)` | optional; duplicate tie-breaker only, never correlated |

## Usage

```bash
python scripts/analysis/mteb_correlation.py --mteb-file results/mteb/leaderboard.csv
python scripts/analysis/mteb_correlation.py --mteb-file results/mteb/leaderboard.csv --new-models-only
```

`--new-models-only` drops the BGE, BERT, RoBERTa, E5 and `text-embedding-*` families.

```bash
python scripts/report_experiments.py mteb
```

`report_experiments.py` reads `results/mteb/leaderboard.csv` by default and is skipped with a note when that file is absent; `--mteb-file` overrides the path.

## Output

Printed to stdout:

- Pearson r and p-value, on the raw score values;
- Spearman rho and p-value, on the same two columns ranked internally;
- the matched-model table with each model's benchmark rank and MTEB retrieval rank.