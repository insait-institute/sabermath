# Per-query latency

Measures average per-query task time for every model in the results tables,
into `results/timing/`.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs
  (see [experiment-evaluation.md](experiment-evaluation.md)).
- API keys for the closed models: `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  and `.openroutertok` at the repo root for `text-embedding-3-*`.
- A GPU node for the neural models. Numbers are only comparable to the
  published table when measured on the same hardware (a single H200) with the
  default 16 CPU lanes.

## Usage

```bash
python scripts/run_timing.py --model rank1-32b
python scripts/run_timing.py --model bm25 --n-queries 30
python scripts/run_timing.py --model gemini-embedding-2
```

Fixed to the **statement-full** task, each model on its production backend, 16
documents in flight per step. The script pins
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=16` before importing numpy;
`SABERMATH_TIMING_THREADS` overrides it, at the cost of comparability.

Build the table with:

```bash
python scripts/report_experiments.py timing
```

## Output

```
results/timing/<model>.json
results/timing/query_sample.json     # the shared query set, so models are comparable
results/tables/RESULTS_timing.md
```

Each `<model>.json` records `mean_seconds`, `median_seconds`, the `backend`
used and the per-query times. Model loading is excluded from the timed region,
and there is no warmup query, so read medians as well as means.

`RESULTS_timing.md` has one row per model: Model, Category, Backend,
Median (s), Mean (s), sorted by median.
