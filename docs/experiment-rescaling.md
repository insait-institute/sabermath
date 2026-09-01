# nDCG rescaling robustness

Replays each stored ranking under a different gain function or relevance scale, to verify whether the benchmark's ordering remains consistent.

## Prerequisites

- `python -m pip install -e .`
- Runs in `results/evaluation/` made with `--save-scores` (on by default).
- `results/rescaling/rankings.npz`.

## Usage

```bash
python scripts/analysis/rescore_ndcg.py --scan results/evaluation \
  --variants linear:1.0 exponent:1.0 exponent:0.6 rank \
  --export-table results/rescaled/summary.json

python scripts/report_experiments.py rescaling      # the LaTeX table
```

| Variant | Meaning |
|---|---|
| `exponent:1.0` | the benchmark's own setting — the control |
| `exponent:0.6` | the "grades in [0,3]" rescale; applied to relevances before the gain |
| `linear:1.0` | linear gain, mathematically scale-invariant |
| `rank` | within-query relevance ranks replace the Bradley-Terry magnitudes |

## Output

```
results/rescaled/summary.json
results/rescaling/results.json
results/tables/rescaling_table.tex
```