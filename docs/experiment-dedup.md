# Deduplication: where does a rephrased copy of a query rank?

Each query's heavily-rephrased copy (`RAG4Math/targets-with-rephrased`,
row-aligned with the queries dataset) is inserted into the corpus, and the
question is where the retriever ranks it.

```bash
python scripts/run_dedup.py                    # every eligible model
python scripts/run_dedup.py --models bge-m3 qwen3-embedding-8b
python scripts/run_dedup.py --models bge-m3 --n 20      # smoke test
```

Results land in `results/dedup/`. Read them with
`python scripts/report_experiments.py dedup`.

## Two regimes, one pass

Both are computed from the same embeddings in one pass, so `--corpus both`
(the default) costs no more than either alone. **They answer different
questions and must never be merged into one table.**

| Regime | What the copy competes against |
|---|---|
| `per-query-candidates` | That query's own 150 candidates — ranked among 151. **The published protocol.** Every model can run it. |
| `all-documents` | All 71,117 documents. Much harder, no published reference. Bi-encoders and lexical models only. |

`all-documents` excludes pair scorers because a cross-encoder or
late-interaction model would need 71,117 forward passes per query.

## Two insertion readings

Each regime reports avg/median rank and top@{1,2,4,…,128} under both:

- `insert_per_query` — only the query's own rephrased copy is in the corpus.
- `insert_all_rephrased` — all 1000 rephrased copies are inserted at once.

## Things that change the numbers

- **Self-matches are excluded by default.** 92.8% of query problems appear
  verbatim among the documents, so leaving them in measures exact-duplicate
  retrieval rather than rephrase retrieval. `--keep-self-match` disables the
  exclusion; the published rows do not use it.
- **`--doc-version full` pairs the rephrased problem with the ORIGINAL
  solution** — the dataset has no rephrased solutions.
- **The lexical rows use different tokenizer configurations from the main
  table**, so a dedup row is not comparable to that model's Table 1 row.

## Checking your setup

Reproduce two published rows before trusting new ones: Octen-8B avg rank
**6.85**, BM25 **24.09**, both under `per-query-candidates` with self-matches
excluded.

## Splitting a slow model

`--query-shards N --query-shard I` gives the same strided split
`run_experiments.py` uses, with each shard writing its own file and
checkpoint. Stitch them back with `python scripts/run_dedup.py
--merge-shards`, which unions the per-query records and **recomputes** every
statistic — the percentiles and top-k coverages are not linear in the query
set, so averaging shard summaries would be wrong even for equal-sized shards.

## Approach Zero

`run_approach0_corpus` does not rank against the whole corpus directly. It
escalates `topk` and re-scores an ambiguous band exactly, through pya0's
per-document `docid=` hook, because `topk=len(documents)` disables Approach
Zero's pruner and costs ~100s per query (~28h for 1,000). Ranks it could not
establish exactly are flagged `rank_is_censored` and are a **lower bound** on
the true rank, never a guess.
