# Is the SABER-Math ranking an artefact of the score rescaling?

Recomputed on the **38 of 42** models in the main results table
(`tab:statement-full`) for which the per-query candidate rankings are stored,
on the **Overall** (statement-full) column, nDCG@10, all 1000 queries.

Reproduce with:

```
python experiments/rescaling_robustness/rescore_rescaling.py --verify
```

## Reproducing this without cluster access

`results/` is gitignored, and the per-query checkpoints it holds are 3.9 GB
across 1319 files (~3.0 MB each), so the study would normally be re-runnable
only on the cluster. It does not have to be. Those files are 3 MB because
they store 17-significant-digit floats for RESUME purposes; nDCG needs only
the ranking, and each ranking is a permutation of 0..149 that fits in a
uint8. The selected runs for all 38 models therefore compress to **5.3 MB**,
committed here as `rankings.npz`:

```
python experiments/rescaling_robustness/rescore_rescaling.py \
    --from-rankings experiments/rescaling_robustness/rankings.npz
```

That reproduces every number below to 1.1e-16, with identical provenance, and
touches nothing outside this directory. Because it keeps all 150 candidates
rather than just the top 10 that nDCG@10 reads, it also supports changing k
(nDCG@20, @50) and any other rank-based metric - MRR, Recall@k - without ever
going back to the 3.9 GB. Truncating to the top 10 would cost 412 KB instead
of 5.3 MB, but locks k at 10; at these sizes that is a bad trade.

Regenerate it with `--export-rankings <path> [--top-k N]`.

## Method

nDCG depends only on a model's *ranking* of the candidates, so no model is
re-run: the rankings dumped by `run_model.py --save-scores` are replayed under
each gain/scale. `--verify` replays all 373 stored runs at the benchmark's own
setting and reproduces the per-query nDCG the original run recorded to
**3.3e-16** — the recomputation is exact, not an approximation.

Each row is built only from a run that reproduces the published Overall score
(worst deviation across the 38 rows: 0.0006, see `results.json`).

## Result

| Setting | Pearson vs orig. | Spearman | Kendall tau | Models changing rank |
| --- | --- | --- | --- | --- |
| [0,3] + exponential gain | 0.9995 | 0.9993 | 0.9915 | 6 / 38 |
| [0,5] + linear gain      | 0.9953 | 0.9982 | 0.9772 | 16 / 38 |
| [0,3] + linear gain      | 0.9953 | 0.9982 | 0.9772 | 16 / 38 |

Scores still correlate near-perfectly. Unlike the earlier 24-model version of
this check, the rank column is no longer empty — the enlarged table packs 42
models into the same score range, so adjacent models sit closer together.

Every one of those changes is nonetheless a **swap of two adjacent models**
(max |delta rank| = 1), and the largest gap between any swapped pair is
**0.0042** nDCG — under half the +-0.008..0.010 95% confidence intervals the
paper itself reports on this column. No swap separates models the benchmark
ever claimed to distinguish; the top of the leaderboard is untouched under
every variant.

Swapped pairs (identical under both linear variants):

| Pair | Original gap | Swaps under |
| --- | --- | --- |
| Qwen3-Reranker-4B / Diver-4B          | 0.0022 | linear |
| Qwen3-Reranker-0.6B / SPLADE-Code-8B  | 0.0021 | linear |
| Rank1-32B / Harrier-27b               | 0.0025 | both |
| Reason-ColBERT / Text-Embedding-3-Large | 0.0003 | both |
| Jina-v5-Nano / Gemma-300m             | 0.0031 | linear |
| BGE-m3 / ReasonIR-8B                  | 0.0042 | linear |
| Harrier-270m / INF-X-Retriever        | 0.0041 | both |
| Jaccard / TF-IDF                      | 0.0006 | linear |

## Two things worth stating in the rebuttal

1. **Linear gain is exactly scale-invariant.** `[0,5] + linear` and
   `[0,3] + linear` are the same estimator; they agree here to 1.1e-16. They
   are two rows in the table, but they are one experiment, and the identical
   numbers are a property of the metric, not a coincidence worth citing as
   independent evidence.
2. **Rescaling moves the absolute numbers a lot** even though it barely moves
   the ranking: max |delta nDCG| is 0.10 for [0,3]+exponential and 0.28 for
   linear gain (linear gain compresses everything upward — RoBERTa goes
   0.311 -> 0.592). The invariance claim is about the *ordering*, not the
   values, and should be worded that way.

## Coverage

Four table rows have no stored candidate rankings, only per-query nDCG, so
they cannot be replayed and are excluded:

| Model | Overall | Why |
| --- | --- | --- |
| ReasonReranker-Qwen3-32B-Rewrite | 0.741 | run without `--save-scores` |
| ReasonEmbed-Qwen3-8B-Rewrite     | 0.738 | run without `--save-scores` |
| ReasonReranker-Qwen3-32B         | 0.733 | run without `--save-scores` |
| ReasonEmbed-Llama-3.1-8B         | 0.664 | run without `--save-scores` |

These four are **queued** (2026-08-30, jobs 780946-780958), one score-capture
run each on the Overall task only:

```
# the two 32B rerankers, 5 strided shards each
for i in 0 1 2 3 4; do
  sbatch --export=ALL,SABERMATH_ENV_NAME=reranker-vllm-retro-star-32b-s$i \
    scripts/rerankers/run_retro_star_32b.slurm \
    --task statement-full --instructions p0 --save-scores --part-name scores \
    --query-shards 5 --query-shard $i
done   # likewise run_retro_star_32b_rewritten.slurm, env ...-rw-s$i

sbatch scripts/rerankers/run_reason_rewriter_reason_embed_8b.slurm \
  --task statement-full --instructions p0 --save-scores --part-name scores
sbatch scripts/rerankers/run_reason_embed_llama_3.1_8b.slurm \
  --task statement-full --instructions p0 --save-scores --part-name scores
```

`--part-name scores` is load-bearing, not cosmetic. Without it the checkpoint
directory is the one the original run already filled, `evaluate_task` skips
every query it finds a stored nDCG for, and the job exits in seconds having
captured no scores at all - the failure mode `_load_checkpoint`'s "raw scores
are unrecoverable without a fresh checkpoint dir" warning describes. The part
suffix forces a fresh directory, so the queries are genuinely recomputed.

`--task statement-full` limits each job to the column this analysis uses,
cutting the bill roughly threefold. The two `-Rewrite` rows read their query
rewrites from the 370 MB log at
`results/rerankers/.rewrites/reason-rewriter-reason-embed-8b.json`, so the 7B
rewriter never loads and only the reranker/retriever pass is paid for.

`run_rerankers.py` now captures scores **by default** (`--no-save-scores`
opts out), on the production path as well as the instruction path, so no
future row can end up unreplayable the way these four did. The old guard that
made `--save-scores` require `--instructions` is gone: it existed only to stop
the flag resuming a checkpoint written before capture, which is now the normal
case, and `evaluate_task` already reports it per run ("resumed queries have no
stored scores").

When the jobs land, just re-run this script: it discovers the new checkpoint
directories automatically, pools the shards by global query row, and admits a
row only if it reproduces that model's published Overall. If a rerun does not
reproduce its table value it will be reported and left out rather than
silently folded in.

One caveat on a row that *is* included: Diver-GroupRank-32B's table value
(0.693) comes from a run whose checkpoint holds no scores. The
protocol-current rerun that does hold them reproduces 0.6924 — a 0.0009
difference, ordinary nondeterminism for a 32B generative reranker. It does not
affect the correlations, which are computed within the replayed set.
