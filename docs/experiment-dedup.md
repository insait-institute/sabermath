# Deduplication

Inserts an LLM-rephrased copy of each query's own problem into the corpus and records where the retriever ranks it. Writes `results/dedup/`.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs (see [experiment-evaluation.md](experiment-evaluation.md)).
- Access to `RAG4Math/targets-with-rephrased`, row-aligned with the queries dataset. Override with `--rephrased`.
- API keys for the closed models, read from the environment: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for `gemini-embedding-*`, `OPENROUTER_API_KEY` for `text-embedding-3-*`.

## Usage

```bash
python scripts/run_dedup.py                              # every eligible model
python scripts/run_dedup.py --models bge-m3 qwen3-embedding-8b
python scripts/run_dedup.py --models bge-m3 --n 20       # smoke test
python scripts/run_dedup.py --merge-shards               # stitch a sharded sweep
```

Options:

| Flag | Effect |
|---|---|
| `--corpus` | `per-query-candidates`, `all-documents`, or `both` (default) |
| `--doc-version full` | pairs the rephrased problem with the original solution |
| `--keep-self-match` | keeps each query's own original document in the corpus |
| `--query-shards N --query-shard I` | strided split, one file per shard |

Build the table with:

```bash
python scripts/report_experiments.py dedup
```

## Output

```
results/dedup/<model>[__all-documents][__shard<i>of<n>].json
results/tables/RESULTS_dedup.md
```
