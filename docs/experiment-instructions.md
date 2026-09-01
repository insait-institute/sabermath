# Instruction prompts

Runs each model under the three instructions and tabulates the change from the no-instruction baseline.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs (see [experiment-evaluation.md](experiment-evaluation.md)).
- The prompt texts in `src/sabermath/instructions.py`:

  | Key | Role |
  |---|---|
  | `p0` | baseline, if not already run |
  | `p1`, `p2`, `p3` | the three instructions |

- API keys for the closed models, read from the environment: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for `gemini-embedding-*`, `OPENROUTER_API_KEY` for `text-embedding-3-*`.

## Usage

```bash
python scripts/run_experiments.py --models <key> --prompts p0 p1 p2 p3
python scripts/report_experiments.py instructions instructions-statement-full
```

For larger models, such as 32B rerankers, it's best to run them sharded and merge the shards for additional parallelism (see [experiment-evaluation.md](experiment-evaluation.md)).

## Output

```
results/evaluation/<model>__<prompt>.json
results/tables/RESULTS_instructions.md
results/tables/RESULTS_instructions_statement_full.md
results/tables/statement_full_instructions.tex
```