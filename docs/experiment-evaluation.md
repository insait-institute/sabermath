# Running an nDCG sweep

`scripts/run_experiments.py` is the only entry point for evaluation. It scores
models across the three tasks (`statement-statement`, `statement-full`,
`full-full`), with per-domain nDCG@10 and the per-query nDCGs the confidence
intervals need, into `results/evaluation/`.

```bash
python scripts/run_experiments.py --list          # every model key
python scripts/run_experiments.py                 # every model, p0: the main tables
python scripts/run_experiments.py --models rank1-7b splade-code-8b
python scripts/run_experiments.py --models rank1-32b --n 20      # smoke test first
```

`--models` defaults to every model in the registry and `--prompts` defaults to
`p0` ("no instruction"), so a bare invocation is the production sweep. `--help`
is the authoritative flag reference; this page covers the parts that are easy
to get wrong.

## Environments

Most models share one environment, `scripts/envs/env_vllm.yml`. Four families
have conflicting pins, will fail loudly in the wrong environment, and need
their own:

| Family | Environment | Why |
|---|---|---|
| `gte-moderncolbert`, `reason-moderncolbert` | `env_colbert.yml` | pylate pins sentence-transformers 5.3.0 |
| `splade-code-*` | `env_splade.yml` | SparseEncoder |
| `inf-retriever-v1-pro`, `inf-x-retriever` | `env_inf_retriever.yml` | bidirectional remote code needs transformers 4.51.x |
| `reasonir-8b` | `env_reasonir.yml` | transformers pinned to the version its remote code was written against |

So in a single environment, run one model or one same-env family per
invocation. Each model runs in its own spawned subprocess regardless, so a CUDA
crash or an OOM never takes down the loop or corrupts results already written.

Each env file's postinstall ends with `pip install -e "$SABERMATH_REPO"`, so
the package's own dependencies come along. See `docs/backend-provenance.md`
for why each pin is load-bearing.

## Resuming

Every task is checkpointed after **each query**, into
`results/evaluation/.checkpoints/<model>/<prompt>__<subset>/<task>.json`.
Re-running the identical command resumes from the last completed query instead
of restarting. This is what makes a 32B generative reranker evaluable at all:
you do not have to guess a large enough time budget up front, because a second
run picks up where the first stopped.

The result file itself is also rewritten from the checkpoints every
`--progress-every` freshly-scored queries (default 10, combined across tasks),
so a hard kill still leaves usable partial results on disk. Lower it for a very
slow model; `--progress-every 0` disables it and writes only at the end.

A partial task entry carries `"n_done"`/`"n_total"`; a finished one does not.
That is the only way to tell a partial write from a complete one, and every
reader in this repo checks it.

## Splitting one run across several jobs

Two independent splits, both of which write isolated files rather than sharing
one — no lock is needed and no writer can lose to another:

```bash
# one job per task
python scripts/run_experiments.py --models rank1-32b --task statement-full \
    --part-name statement-full

# strided query shards (i % N == I), so every shard sees a mix of domains
python scripts/run_experiments.py --models rank1-32b --query-shards 4 --query-shard 0
```

Stitch the shards back together with the same endpoint:

```bash
python scripts/run_experiments.py --merge-shards                 # every group in results/evaluation
python scripts/run_experiments.py --merge-shards results/evaluation/rank1-32b__p0__shard*of4.json
python scripts/run_experiments.py --merge-shards --dry-run       # report, write nothing
```

The merge places each shard's per-query nDCG at its **global** query index and
recomputes the overall and per-domain means from the reassembled array. It
never averages the shards' own means — the stride leaves shards differing by
one, so averaging would be subtly wrong. It refuses to write a merge with
queries missing from every part unless you pass `--allow-incomplete`.

`scripts/run_dedup.py` has the same `--query-shards` / `--merge-shards` pair.

## Output naming

```
results/evaluation/<model>__<prompt>[__<subset>][__part-<name>].json
```

Running a single `--task` merges into that model's file task by task rather
than overwriting it, and re-running a task replaces just that task's entry. A
crashed run never erases good data already saved for the same file.

A `--n`/`--seed` subset writes to a *different*, suffixed file
(`__n20_seed42`), so a smoke test can never collide with or be merged into a
full run.

## After a sweep

```bash
python scripts/report_experiments.py
```

This recomputes the bootstrap confidence intervals first, then regenerates
every table into `results/tables/`. It is safe to run at any point: a
generator whose inputs are missing is skipped with a note, and a
partially-finished sweep produces a partial table rather than a wrong one.
