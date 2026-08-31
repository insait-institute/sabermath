# Confidence intervals

Bootstrap confidence intervals for SABER-Math retrieval results.

The confidence-interval code takes query-level nDCG values, resamples them by
benchmark domain and overall task, and writes one JSON result file per
retrieval method into `results/confidence/`. A second step converts those into
LaTeX tables.

```bash
python scripts/report_experiments.py confidence     # recompute from results/evaluation
python -m sabermath.reporting.json_to_tex           # format them as LaTeX
```

`scripts/report_experiments.py` runs the recompute before every table, so the
intervals the main tables print are never stale.

This page describes the experiment by purpose rather than by paper table,
figure, or section number, since paper numbering may change.

## What the experiment does

For each retrieval method, `sabermath.analysis.confidence` evaluates
query-level nDCG values and repeatedly resamples them to estimate uncertainty
around the reported scores. By default it covers the main retrieval setting and
reports both overall and domain-level intervals.

To recompute from an existing run rather than re-scoring a model:

```bash
python -m sabermath.analysis.compute_confidence_intervals results/evaluation/<model>__p0.json
```

The bootstrap uses:

- 10,000 bootstrap samples;
- fixed random seed `42411`;
- 300 sampled queries per domain for domain-level estimates;
- 95% percentile intervals using the 2.5th and 97.5th percentiles.

## Expected caches

`confidence.py` re-scores a model, so it wants the precomputed embedding/vector
caches for dense or API-based models. `compute_confidence_intervals` needs no
model at all - it resamples the per-query nDCGs already stored in a run file,
which is why `report_experiments.py` uses that path.

The cache directory defaults to the repo root:

```bash
.vector.cache