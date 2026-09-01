# Instruction prompts

Runs each model under four instructions and tabulates the change from the
no-instruction baseline. The runs go to `results/evaluation/` alongside every
other nDCG run.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs
  (see [experiment-evaluation.md](experiment-evaluation.md)).
- The prompt texts are `src/sabermath/instructions.py`:

  | Key | Role |
  |---|---|
  | `p0` | no instruction — the baseline |
  | `p1`, `p2`, `p3` | the three instructions |
  | `pm` | the repo's production math instruction; a separate instruction, not part of the ablation |

- Models with no instruction mechanism (`approach0`, `bm25`, `jaccard`,
  `tf-idf`) are listed in `INSTRUCTION_EXCLUDED` in
  `src/sabermath/registry.py` and are refused with an error if you ask for
  `p1`–`p3`. Models whose mechanism is a fixed prompt grammar rather than free
  text are in `INSTRUCTION_CONTROL_REASONS` and are reported as controls.

## Usage

```bash
python scripts/run_experiments.py --models <key> --prompts p0 p1 p2 p3
python scripts/report_experiments.py instructions instructions-statement-full
```

Several instructions in one invocation share the loaded model. Reporting needs no GPU
and no downloads.

A 32B reranker will not finish 1000 queries in one sitting; run it sharded and
merge (see [experiment-evaluation.md](experiment-evaluation.md)). The reporter
pools shards either way.

## Output

```
results/evaluation/<model>__<prompt>.json
results/tables/RESULTS_instructions.md
results/tables/RESULTS_instructions_statement_full.md
results/tables/statement_full_instructions.tex
```

`RESULTS_instructions.md` gives one row per model and one column per instruction, with
the parenthesised value being the change from that model's own `p0`.
Instructable models and controls are separate blocks, and the `pm` instruction has its
own section.

A cell appears only when all 1000 queries scored; a partial run is listed
under "Incomplete" rather than averaged over its non-null entries.

Where a model has both an explicit `__p0` file and a bare `<model>.json`, the
explicit one is used. For the Qwen3-Reranker family the two differ, so this
table's `p0` for that family is lower than the main table's.
