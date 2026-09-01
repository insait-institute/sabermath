# Per-query latency

Measures average per-query task time for every model in the results tables, into `results/timing/`.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs (see [experiment-evaluation.md](experiment-evaluation.md)).
- API keys for the closed models: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for `gemini-embedding-*`, `OPENROUTER_API_KEY` for `text-embedding-3-*`. Every other model runs locally and needs none.
- A GPU for the neural models.

## Usage

Sample the shared query set once, before any per-model job:

```bash
python scripts/run_timing.py --generate-sample --n-queries 30
```

Then run the models. Every one reads that file, so `--n-queries`/`--seed` are ignored here, you can change them by re-running `--generate-sample` and then every model again.

```bash
python scripts/run_timing.py --model rank1-32b
python scripts/run_timing.py --model bm25
python scripts/run_timing.py --model gemini-embedding-2
```

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
