# Confidence intervals

Bootstrap confidence intervals over the per-query nDCGs a run already stored.

## Prerequisites

- `python -m pip install -e .`
- Finished runs in `results/evaluation/` carrying `ndcgs_by_task`. No model,
  GPU or vector cache is needed — nothing is re-scored.

## Usage

```bash
python scripts/report_experiments.py confidence     # every p0 run
python scripts/analysis/compute_confidence_intervals.py results/evaluation/<model>__p0.json
```

`report_experiments.py` runs this before every table, so the intervals the
tables print are never stale. It selects inputs with
`sabermath.results.load_runs` — one p0 run per model, no shards, parts or
`--n` subsets. Calling the script by hand, exclude the smoke-test files
(`*__n20_seed42.json`) yourself.

The protocol is fixed:

| Setting | Value |
|---|---|
| Bootstrap samples | 10,000 |
| Seed | 42411, a fresh RNG per (model, task) |
| Domain resample | 300 queries per domain |
| Overall resample | full query count |
| Interval | 95% percentile (2.5th / 97.5th) |
| Draw order | each domain in DOMAINS order, then overall — part of the result under a fixed seed |

Queries whose nDCG is still null are skipped with a warning.

`scripts/analysis/recompute_confidence.py` is the alternative that re-scores a
model instead of reading a run; it needs the vector cache, which defaults to
`.vector.cache` at the repo root.

## Output

```
results/confidence/<model>.json
results/tables/RESULTS_confidence_intervals.md
```

Each JSON is `{"k": 10, "tasks": [...]}`, one entry per task with `task`,
`mean`, `confidence_interval` as `[lo, hi]`, and `branches[]` giving the same
pair per domain.
