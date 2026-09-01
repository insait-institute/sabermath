# nDCG evaluation

Scores models across the three tasks and writes one JSON run per
(model, prompt) into `results/evaluation/`.

## Prerequisites

- `python -m pip install -e .` — every script imports the installed package.
- The environment that model family needs. Most models share
  `scripts/envs/env_vllm.yml`; four families have conflicting pins and fail
  loudly in the wrong one:

  | Family | Environment |
  |---|---|
  | `gte-moderncolbert`, `reason-moderncolbert` | `env_colbert.yml` |
  | `splade-code-*` | `env_splade.yml` |
  | `inf-retriever-v1-pro`, `inf-x-retriever` | `env_inf_retriever.yml` |
  | `reasonir-8b` | `env_reasonir.yml` |

  So run one model, or one same-env family, per invocation.
- API keys for the closed models: `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## Usage

```bash
python scripts/run_experiments.py --list                        # every model key
python scripts/run_experiments.py                               # every model, p0
python scripts/run_experiments.py --models rank1-7b splade-code-8b
python scripts/run_experiments.py --models rank1-32b --n 20     # 20-query smoke test
python scripts/run_experiments.py --prompts p0 p1 p2 p3         # instruction arms
```

`--models` defaults to every model in the registry, `--prompts` to `p0`.
`--help` is the authoritative flag reference.

Re-running an identical command resumes from the last completed query.

### Splitting one run across jobs

```bash
# one job per task
python scripts/run_experiments.py --models rank1-32b --task statement-full \
    --part-name statement-full

# strided query shards
python scripts/run_experiments.py --models rank1-32b --query-shards 4 --query-shard 0

# stitch the shards back
python scripts/run_experiments.py --merge-shards                 # scan results/evaluation
python scripts/run_experiments.py --merge-shards <part>.json ... # an explicit group
python scripts/run_experiments.py --merge-shards --dry-run       # report, write nothing
```

`--merge-shards` refuses to write when queries are missing from every part;
pass `--allow-incomplete` to override. `scripts/run_dedup.py` takes the same
pair of flags.

### After a sweep

```bash
python scripts/report_experiments.py
```

## Output

```
results/evaluation/<model>__<prompt>[__<subset>][__part-<name>].json
results/evaluation/.checkpoints/<model>/<prompt>__<subset>/<task>.json
```

Each run file holds `domains` and `reports.<model>` with:

| Field | Contents |
|---|---|
| `model`, `processor`, `dcg_variant`, `k` | how it was scored |
| `prompt` | the arm, its template and the applied input envelope |
| `tasks[]` | per task: `ndcg_at_k` and per-domain `branches[]` |
| `ndcgs_by_task` | the per-query nDCG list the confidence intervals read |
| `n_done` / `n_total` | **present only on an incomplete task** |

A `--n`/`--seed` subset writes a suffixed file (`__n20_seed42`) and can never
merge into a full run. Running a single `--task` merges into the model's file
task by task rather than overwriting it.
