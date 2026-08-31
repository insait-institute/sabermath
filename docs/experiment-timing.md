# Per-query latency

`scripts/run_timing.py` measures a standardized average per-query task time
for every model in the results tables — rerankers, bi-encoder embedders (open
and closed/API) and lexical baselines — into `results/timing/`.

```bash
python scripts/run_timing.py --model rank1-32b
python scripts/run_timing.py --model bm25 --n-queries 30
python scripts/run_timing.py --model gemini-embedding-2
```

The task is fixed to **statement-full** — a bare problem statement as the
query, problem+solution as the document — the paper's main setting, so a
latency number lines up with an nDCG number.

## Methodology

**1. Production backends only.** Every model is timed on the exact processor
construction `run_experiments.py` uses; the reranker-pipeline models are built
through the registry's own builder tables, so the two cannot drift apart. A
latency number is therefore never measured on a backend the benchmark does not
score with.

**2. Standardized parallelism: 16 documents in flight per step**, rather than
each model's own tuned default (`BATCH_SIZE`). Concretely:

| Family | How the 16 is applied |
|---|---|
| vLLM models | Candidates submitted in slices of 16 per embed/score/generate call |
| `rank1-*` | The production single generate over ~150 candidates is sliced into 16-prompt calls |
| `diver-grouprank-32b` | `group_size=20` is the model's official *scoring* protocol, not a batching knob — changing it changes scores, so only the generate calls over group-prompts are sliced |
| HF/SentenceTransformers (splade, colbert, jina-v5-nano) | A true `batch_size=16` |
| `text-embedding-3-*` | No batch API, so `max_concurrency=16` single-document requests |
| `gemini-embedding-001` | One 16-document request at a time |
| `gemini-embedding-2` | The API forces one document per request, so 16 concurrent single-document requests |
| Lexical baselines | 16-document slices against an index fit on the FULL candidate set, so the scores are identical and corpus statistics are never chunked |

Caveats worth stating with the numbers: vLLM still micro-batches within a
slice, so this caps request-level parallelism rather than imposing lockstep
batches. `rank1-*` reasoning-chain lengths vary wildly, so each slice waits on
its slowest member and sliced timing reads *worse* than production's one big
call. And a provider's server-side batching is opaque — only the client side
is standardized.

`approach0` is the one genuine exception: it delegates to pya0's search
engine, which exposes no per-document scoring hook to slice, so it runs
unmodified. Per-document worker threads were considered and rejected for the
other lexical methods — jaccard is pure-Python set arithmetic where the GIL
serializes threads, bm25/tf-idf per-document scoring is microseconds of
vectorized work where thread overhead would dominate, and either way it would
time an implementation production never runs.

**3. CPU thread cap: 16 lanes.** The script sets
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=16` before importing numpy — the
CPU-side counterpart of the GPU batch budget, applied uniformly, so no
BLAS-backed method quietly uses every core on the machine.
`SABERMATH_TIMING_THREADS` overrides it, at the cost of comparability with the
published numbers.

**4. No amortization for embedders.** Every sampled query is scored from a
cold cache (`check_cache=False`, `update_cache=False`), so no candidate
document embedding is reused across queries.

**5. Model loading is excluded** from the timed region: only the time from
"start scoring this query's candidates" to "have the final sorted ranking" is
measured, per query. There is no warmup query, so a model whose first call
pays a one-time JIT or CUDA-graph cost shows that in its numbers — read
medians as well as means.

**6. One machine.** Everything was measured on a single H200 node, including
the lexical and closed-API models that need no GPU, for environment
consistency. A number measured on different hardware is internally consistent
but not comparable to the published table.

## One shared query sample

Every model reads the same `query_sample.json`, so a latency table is never a
comparison across different query sets.
