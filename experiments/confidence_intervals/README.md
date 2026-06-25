# Confidence intervals

This directory computes bootstrap confidence intervals for SABER-Math retrieval results.

The confidence-interval script uses query-level nDCG values, resamples them by benchmark domain and overall task, and writes one JSON result file per retrieval method. The table-formatting script then converts those JSON files into LaTeX tables.

The README describes the experiment by purpose rather than by fixed paper table, figure, or section numbers, since paper numbering and result labels may change.

## What the experiment does

For each retrieval method, `confidence.py` evaluates query-level nDCG values and repeatedly resamples them to estimate uncertainty around the reported scores. By default, it computes confidence intervals for the main retrieval setting and reports both overall and domain-level intervals.

It runs directly by:

```
python confidence.py
```

The script uses:

- 10,000 bootstrap samples;
- fixed random seed `42411`;
- 300 sampled queries per domain for domain-level estimates;
- 95% percentile intervals using the 2.5th and 97.5th percentiles.

## Expected caches

The script is designed to run from precomputed embedding/vector caches for dense or API-based retrieval models.

By default, it expects the cache directory here:

```bash
../../.vector.cache