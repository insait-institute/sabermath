# nDCG evaluation

Scores models across the three tasks and writes one JSON run per (model, prompt) into `results/evaluation/`.

## Prerequisites

- `python -m pip install -e .` — every script imports the installed package.
- The environment that model family needs. Most models share `scripts/envs/env_vllm.yml`, except for:

  | Family | Environment |
  |---|---|
  | `gte-moderncolbert`, `reason-moderncolbert` | `env_colbert.yml` |
  | `splade-code-*` | `env_splade.yml` |
  | `inf-retriever-v1-pro`, `inf-x-retriever` | `env_inf_retriever.yml` |
  | `reasonir-8b` | `env_reasonir.yml` |

- API keys for the closed models, read from the environment: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for `gemini-embedding-*`, `OPENROUTER_API_KEY` for `text-embedding-3-*`.

## Usage

```bash
python scripts/run_experiments.py --list                        # every model key
python scripts/run_experiments.py                               # every model, p0
python scripts/run_experiments.py --models rank1-7b splade-code-8b
python scripts/run_experiments.py --models rank1-32b --n 20     # 20-query smoke test
python scripts/run_experiments.py --prompts p0 p1 p2 p3         # three instructions + baseline
```

### Splitting one run across jobs

```bash
# one job per task
python scripts/run_experiments.py --models rank1-32b --task statement-full \
    --part-name statement-full

# strided query shards
python scripts/run_experiments.py --models rank1-32b --query-shards 4 --query-shard 0

# stitch the shards back
python scripts/run_experiments.py --merge-shards
python scripts/run_experiments.py --merge-shards <part>.json ...
python scripts/run_experiments.py --merge-shards --dry-run
```

`--merge-shards` refuses to write when queries are missing from every part; pass `--allow-incomplete` to override. `scripts/run_dedup.py` takes the same pair of flags.

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
| `prompt` | the instruction, its template and the applied input envelope |
| `tasks[]` | per task: `ndcg_at_k` and per-domain `branches[]` |
| `ndcgs_by_task` | the per-query nDCG list the confidence intervals read |
| `n_done` / `n_total` | **present only on an incomplete task** |

A `--n`/`--seed` subset writes a suffixed file (`__n20_seed42`) and can never merge into a full run. Running a single `--task` merges into the model's file task by task rather than overwriting it.
