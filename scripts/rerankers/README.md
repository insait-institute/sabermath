# Reranker sweep: SLURM launchers

Runs the 7 SABER-Math models that need a custom `ModelProcessor`
(`rank1-32b`, `qwen3-embedding-8b`, `qwen3-reranker-8b`, `gte-moderncolbert`,
`reason-moderncolbert`, `reasonir-8b`, `splade-code-8b`) across all 3 tasks
(`statement-statement`, `statement-full`, `full-full`), with per-domain nDCG
and per-query nDCGs (for 95% CIs) saved to `results/rerankers/*.json`.

See `scripts/run_rerankers.py`'s module docstring for the underlying driver
and why each model gets its **own conda env** (some have conflicting pins,
e.g. ReasonIR vs. Qwen3-Reranker's transformers requirement) and its **own
SLURM job** (so one model failing/timing out never blocks the others, and
each can be resubmitted independently).

## One-time setup

1. **HF token.** `scripts/rerankers/_common.sh` picks one up automatically -
   no file needs to be created for this - checking in order:
   1. `.hftok` in the sabermath repo root, if present (explicit override -
      not committed; add it to `.gitignore` if it isn't already covered).
   2. `~/.cache/huggingface/token`, i.e. whatever `huggingface-cli login`
      already left on this machine.

   If neither exists, `_common.sh` prints a warning and the job proceeds
   without a token (fine for ungated models, will fail to download
   rank1-32b/ReasonIR-8B/SPLADE if they're gated for your account). Do not
   reuse the token that was previously hardcoded in
   `rag-math-test/rank-embedding-math/running-rerankers/rank1.sh` - that one
   is exposed in plaintext and should be rotated separately regardless.

2. **Log directory** (SLURM needs `--output`'s directory to exist *before*
   the job starts):
   ```bash
   mkdir -p /scratch/$USER/logs
   ```

## Running

From the sabermath repo root:

```bash
# Everything - all 7 models, all 3 tasks, as 7 independent jobs
bash scripts/rerankers/submit_all.sh

# One model
sbatch scripts/rerankers/run_splade_code_8b.slurm

# Smoke test one model on a random 20-query subset before committing to a
# full ~1000-query run (recommended for rank1-32b especially)
sbatch scripts/rerankers/run_rank1_32b.slurm --n 20

# Just one task
sbatch scripts/rerankers/run_reasonir_8b.slurm --task statement-full
```

Check status: `squeue -u $USER`. Logs land in
`/scratch/$USER/logs/reranker_<model>.log` (`.error` for stderr).

**Cluster region.** This cluster spans multiple regions (`zone-sof1`/
`zone-gcp-eu1` in the EU, `zone-msp3` in the US) with **separate, non-synced
home-directory storage per region** - `msp3`'s copy of this repo turned out
to be a months-old stub, so jobs landing there used to fail immediately
(`cd`/`source` on a path that "existed" but was actually a different,
near-empty directory - confirmed by direct diagnostics, not a script bug).

Rather than avoid `msp3` (it's often the least busy region, per `sinfo`),
every `run_*.slurm` now **self-heals**: right after `cd`-ing into the repo,
unless already on the canonical host, it `rsync`s the repo over SSH from
there (`hala`, hardcoded as `$SABERMATH_CANONICAL_HOST` - override via that
env var if the canonical host ever changes) before doing anything else, and
sets `NEEDS_REGION_SYNC=1`. This pull uses `-u`/update mode and is **not**
gated on "is `_common.sh` missing" - a presence check isn't enough (a node
can have a stale-but-present copy from an earlier sync, which happened) - so
it always re-syncs, cheaply, and it includes `results/`/`.checkpoints/` too
(update mode means it only fills gaps or takes the newer side per file,
never regresses a checkpoint that's further along locally already - e.g.
resubmitting on the same node keeps that node's own progress).

Two independent things push progress back to the canonical copy, so results
from a non-canonical run are never stranded where
`compute_confidence_intervals.py` (run from `hala`) can't see them:
1. **Periodic, from Python** - `scripts/run_rerankers.py`'s `on_progress`
   callback, every `--progress-every` freshly-scored queries (default 10,
   combined across all tasks in the run - see next section). This is the
   main defense, since it doesn't depend on the process getting a chance to
   clean up at all.
2. **On exit, from bash** - `_common.sh` arms an `EXIT` trap
   (`sync_back_to_canonical`, only when `NEEDS_REGION_SYNC=1`) as a
   last-call safety net for whatever happened since the last periodic push.
   Only fires on a normal exit or a caught signal like SIGTERM (what SLURM
   sends on a wall-clock timeout - past rank1-32b logs show a clean shutdown
   on receipt of it); a bare `SIGKILL` skips it, which is exactly why (1)
   exists - the periodic push means a `SIGKILL` loses at most
   `--progress-every` queries, not the whole run.

Verified end-to-end against `msp3-1` directly (pull, push, and the periodic
on_progress path with a mocked rsync call), not just read through.

**GPU count.** Every job requests `--gpus=h200:2`. Only **rank1-32b** and
**qwen3-embedding-8b** actually shard across both (vLLM
`tensor_parallel_size`, set automatically from `$SLURM_GPUS_ON_NODE` - see
`run_rank1_32b.slurm`/`run_qwen3_embedding_8b.slurm`). The other 5 models
(`qwen3-reranker-8b`, `gte-moderncolbert`, `reason-moderncolbert`,
`reasonir-8b`, `splade-code-8b`) are single-process HF/PyLate calls that
don't shard in this framework, so their second GPU sits idle - harmless if
your scheduler/allocation doesn't mind, but trim their `--gpus=h200:2` down
to `--gpus=h200:1` if you'd rather not reserve GPUs nothing will use.

## If a job times out (especially rank1-32b)

Every model/task is checkpointed to
`results/rerankers/.checkpoints/<model>/<subset>/<task>.json` after *every*
query. If a job is killed by its wall-clock limit, just resubmit the exact
same command - it resumes from the last completed query instead of starting
over. This is what actually solves the failure mode that killed 6 of 7 past
rank1-32b attempts (see `rag-math-test/rank-embedding-math/running-rerankers/`
for that history): you no longer need to guess a large-enough `--time` up
front, since a second (or third) submission just picks up where the last one
left off.

## If a job gets killed with no warning (root cancellation, node failure, OOM, ...)

Checkpointing (above) already protects the *internal* per-query state -
resuming reads it back regardless. Two more things make an *unexpected* kill
(not a graceful timeout) survivable too:

- `results/rerankers/<model>.json` itself - the file `compute_confidence_intervals.py`
  actually reads - gets rewritten from the checkpoints every
  `--progress-every` freshly-scored queries (default 10; combined across all
  tasks in one run, not per-task), not only once at the very end. So a kill
  mid-run still leaves usable, if partial, results on disk - look for
  `"n_done"`/`"n_total"` on each task entry to tell a partial write from a
  finished one.
- If the job is on a non-canonical region (`msp3` today), that same
  `--progress-every` cadence also pushes `results/` back to `hala` (see
  "Cluster region" above) - so even a bare `SIGKILL` there loses at most
  `--progress-every` queries' worth of progress, not the whole run.

Tune the cadence with `--progress-every N` (passed after the script name,
same as `--n`/`--task`):
```bash
sbatch scripts/rerankers/run_rank1_32b.slurm --progress-every 5   # tighter safety net, more rsync overhead
sbatch scripts/rerankers/run_splade_code_8b.slurm --progress-every 0  # disable - only write at the end, the old behavior
```
Lower values mean less to lose on a hard kill, at the cost of more frequent
`rsync` calls (each blocks the query loop until it completes or times out
after 120s) - the fast, non-`msp3` case is a local-disk write, closer to
free; the cross-region case is a real network round-trip, so don't set this
very low for a fast model (e.g. the ColBERT variants finish ~150-candidate
queries quickly enough that `--progress-every 10` could mean an rsync call
every few seconds).

## After all jobs finish

```bash
python testing/compute_confidence_intervals.py results/rerankers/*.json
```

This writes `testing/confidence_intervals.json` with, per model/task
(including `statement-full`) and per domain: mean nDCG@10, normal 95% CI,
and bootstrap 95% CI - reusing the exact script that already computes this
for the lexical baselines (BM25/TF-IDF/Jaccard), now merged with the
reranker results too.

## Result file naming and merging

Each model's full-dataset results live at `results/rerankers/<model>.json`.
Running just one `--task` for a model (e.g. to split tasks across separate
jobs) merges into that same file task-by-task instead of overwriting it -
`statement-full` from one submission and `full-full` from a later one both
end up in the same file, and re-running a task you already have just
replaces that task's entry. A crashed run never erases previously-saved good
data for the same file.

A `--n`/`--seed` subset (smoke test) writes to a *different*,
suffixed file - `results/rerankers/<model>__n<N>_seed<seed>.json` - so it
can never collide with or get merged into the full run. Keep that in mind
when globbing into `compute_confidence_intervals.py`: pass
`results/rerankers/*.json` once you've deleted or moved aside any smoke-test
files, or list the exact full-run files explicitly, so a leftover
`__n20_seed42` file doesn't get merged in.
