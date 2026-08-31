# nDCG rescaling robustness

Does the benchmark's ranking of models survive a different gain function or a
different relevance scale? nDCG depends only on a model's *ranking* of the
candidates, so this needs no model re-run: the stored rankings are replayed
under each variant.

```bash
python -m sabermath.analysis.rescore_ndcg --scan results/evaluation \
  --variants linear:1.0 exponent:1.0 exponent:0.6 rank \
  --export-table results/rescaled/summary.json

python scripts/report_experiments.py rescaling      # the LaTeX table
```

## What gets stored, and why the ranking and not just the scores

`--save-scores` (on by default) persists, per query, the raw 150 candidate
scores **and the ranking actually used**, to
`<checkpoint-dir>/<task>.scores.json` beside a `meta.json` join key.

The ranking is stored rather than re-derived because `argsort` is unstable on
ties: re-sorting the same scores can produce a different order, and on ties
that changes nDCG. Replaying the stored ranking is what makes the
recomputation exact rather than approximate.

## The variants

| Variant | Meaning |
|---|---|
| `exponent:1.0` | The benchmark's own setting — the control |
| `exponent:0.6` | The "grades in [0,3]" rescale; the scale is applied to relevances BEFORE the gain |
| `linear:1.0` | Linear gain. Mathematically scale-invariant (gains cancel in the IDCG ratio), so one linear column suffices |
| `rank` | Rank-only gains: within-query relevance ranks replace the Bradley–Terry magnitudes, linear gain |

The variant parser rejects `exponential` explicitly. That exact typo silently
killed an earlier rescaling run, so it is now a loud error rather than a
skipped variant.

## Self-checks

Every run asserts, and fails loudly:

- recomputed `exponent:1.0` equals the stored per-query nDCGs to **1e-12**;
- `linear` is scale-invariant.

Tie instability between the stored rankings and re-argsorted scores is
reported informationally rather than as a failure.

Parts and shards of the same cell are pooled by **global query row** before
averaging, so the rescaled table is unaffected by how a run was split across
jobs.

`--verify` replays all stored runs at the benchmark's own setting and
reproduces the per-query nDCG each run recorded to 3.3e-16. Each row of the
published table is built only from a run that reproduces its published Overall
score; the worst deviation across the 42 rows is 0.0006. See
`results/rescaling/RESULTS.md` for the numbers and their provenance.
