# nDCG rescaling robustness

Replays each stored ranking under a different gain function or relevance
scale, to test whether the benchmark's ordering of models survives.

## Prerequisites

- `python -m pip install -e .`
- Runs in `results/evaluation/` made with `--save-scores` (on by default),
  which persists per query the raw candidate scores **and the ranking actually
  used**. The ranking is stored rather than re-derived because `argsort` is
  unstable on ties.
- `results/rescaling/rankings.npz` for the LaTeX table step.

No model is re-run: nDCG depends only on the ranking.

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
| `rank` | within-query relevance ranks replace the Bradley–Terry magnitudes |

`exponential` is rejected as a variant name rather than silently skipped.
`--verify` replays every stored run at the benchmark's own setting instead of
producing a table.

Parts and shards of the same cell are pooled by global query row before
averaging, so the result does not depend on how a run was split.

## Output

```
results/rescaled/summary.json
results/tables/rescaling_table.tex
```

Every run asserts, and fails loudly, that recomputed `exponent:1.0` equals the
stored per-query nDCGs to 1e-12 and that `linear` is scale-invariant. Tie
instability between the stored ranking and a re-argsort of the scores is
reported informationally.
