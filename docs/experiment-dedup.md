# Deduplication

Inserts an LLM-rephrased copy of each query's own problem into the corpus and
records where the retriever ranks it. Writes `results/dedup/`.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs
  (see [experiment-evaluation.md](experiment-evaluation.md)).
- Access to `RAG4Math/targets-with-rephrased`, row-aligned with the queries
  dataset. Override with `--rephrased`.

## Usage

```bash
python scripts/run_dedup.py                              # every eligible model
python scripts/run_dedup.py --models bge-m3 qwen3-embedding-8b
python scripts/run_dedup.py --models bge-m3 --n 20       # smoke test
python scripts/run_dedup.py --merge-shards               # stitch a sharded sweep
```

Options that change what is measured:

| Flag | Effect |
|---|---|
| `--corpus` | `per-query-candidates`, `all-documents`, or `both` (default) |
| `--doc-version full` | pairs the rephrased problem with the ORIGINAL solution |
| `--keep-self-match` | keeps each query's own original document in the corpus |
| `--query-shards N --query-shard I` | strided split, one file per shard |

`approach0` is not supported: its crash skip-list is keyed to raw benchmark
query text.

Build the table with:

```bash
python scripts/report_experiments.py dedup
```

## Output

```
results/dedup/<model>[__all-documents][__shard<i>of<n>].json
results/tables/RESULTS_dedup.md
```

Each file holds `model`, `regime`, `n_queries`, `n_self_matches_excluded`,
`per_query`, and two summaries:

| Key | Reading |
|---|---|
| `insert_per_query` | only the query's own rephrased copy is in the corpus |
| `insert_all_rephrased` | all 1000 rephrased copies inserted at once (`null` for pair scorers) |

Each summary carries `avg_rank`, `median_rank`, `trimmed_mean_rank_95`,
`p90_rank`, `p99_rank`, `max_rank` and `top_{1,2,4,…,128}`. **Lower rank is
better.** Read the median and the top-k coverage; `avg_rank` has a long tail.

`RESULTS_dedup.md` gives one table per regime — Model, SABER-Math
(statement-statement nDCG@10), Avg. Rank, Med. Rank, Top-1…Top-128.

Reference values, `per-query-candidates` with self-matches excluded:
Octen-8B avg rank **6.85** / median **2.0**; `bm25` **26.98**;
`bm25-no-tok` **24.02**.
