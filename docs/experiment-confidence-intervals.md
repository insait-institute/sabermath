# Confidence intervals

Bootstrap confidence intervals for SABER-Math retrieval results.

The confidence-interval code takes query-level nDCG values, resamples them by
benchmark domain and overall task, and writes one JSON result file per
retrieval method into `results/confidence/`. A second step converts those into
LaTeX tables.

```bash
python scripts/report_experiments.py confidence     # recompute from results/evaluation
python scripts/report_experiments.py           # format them as LaTeX
```

`scripts/report_experiments.py` runs the recompute before every table, so the
intervals the main tables print are never stale.

This page describes the experiment by purpose rather than by paper table,
figure, or section number, since paper numbering may change.

## What the experiment does

For each retrieval method, `scripts/analysis/recompute_confidence.py` evaluates
query-level nDCG values and repeatedly resamples them to estimate uncertainty
around the reported scores. By default it covers the main retrieval setting and
reports both overall and domain-level intervals.

To recompute from an existing run rather than re-scoring a model:

```bash
python scripts/analysis/compute_confidence_intervals.py results/evaluation/<model>__p0.json
```

The protocol is fixed, so new models' intervals stay directly comparable with
the published ones:

- 10,000 bootstrap samples;
- fixed random seed `42411`;
- 300 sampled queries per domain for domain-level estimates;
- 95% percentile intervals using the 2.5th and 97.5th percentiles;
- a fresh RNG per (model, task), and a fixed **draw order** — each domain in
  turn, then the full-size overall resample. With a fixed seed the order is
  part of the result, so reordering those loops changes every interval;
- queries whose nDCG is still null (an unfinished checkpoint) are skipped,
  with a loud warning, so a partial run is never mistaken for a finished one.

Exclude smoke-test files (`results/evaluation/*__n20_seed42.json`) from the
glob: their checkpoints cover a 20-query subset, not the benchmark.

## Expected caches

`confidence.py` re-scores a model, so it wants the precomputed embedding/vector
caches for dense or API-based models. `compute_confidence_intervals` needs no
model at all - it resamples the per-query nDCGs already stored in a run file,
which is why `report_experiments.py` uses that path.

The cache directory defaults to the repo root:

```bash
.vector.cache