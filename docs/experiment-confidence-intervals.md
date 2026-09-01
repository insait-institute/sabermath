# Confidence intervals

Bootstrap confidence intervals over the per-query nDCGs a run already stored.

## Prerequisites

- `python -m pip install -e .`
- Finished runs in `results/evaluation/` carrying `ndcgs_by_task`.

## Usage

```bash
python scripts/report_experiments.py confidence
python scripts/analysis/compute_confidence_intervals.py results/evaluation/<model>__p0.json
```

`report_experiments.py` runs this before every table, so the intervals the tables print are never stale. It selects inputs with `sabermath.results.load_runs`, one run per model.

Queries whose nDCG is still null are skipped with a warning.

`scripts/analysis/recompute_confidence.py` is the alternative that re-scores a model instead of reading a run.

## Output

```
results/confidence/<model>.json
results/tables/RESULTS_confidence_intervals.md
```
