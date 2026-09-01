# Construction-pipeline checks

Two checks on how the benchmark's relevance scores were produced.

## Prerequisites

- `python -m pip install -e .` (needs `pandas`, `scikit-learn`, `matplotlib`,
  `seaborn`).
- `scripts/config/tournament.yaml` names the datasets and prompt files. Pass
  `--config-file` to use a different one.
- An LLM API for the tournament simulation; credentials as at the repo root
  (`.geminitok`, `.openroutertok`) or the matching environment variables.

## Usage

### Tournament convergence

Simulates a Swiss tournament and counts, every 5 rounds, the inversions
between its ordering and a full round-robin scored with Bradley-Terry.

```bash
python scripts/analysis/inversions.py \
    --output-json final_results.json \
    --matches-save-file matches_res_inv.csv \
    --num-rounds 50

python scripts/plots/plot_inversions.py --results-json final_results.json
```

The run checkpoints into `--matches-save-file` every `--checkpoint-every`
rounds and resumes from it; `--force-recalc` ignores an existing checkpoint.

### Selection-signal effect

```bash
python scripts/analysis/signal_effect.py
```

Restrict to one domain with `--algebra`, `--geometry`, `--combinatorics`,
`--calculus` or `--number-theory`.

## Output

| Path | Contents |
|---|---|
| `--output-json` (default `final_results.json`) | `average_num_inversions_after_<n>_rounds` per checkpointed round |
| `--matches-save-file` | every pairwise match played, one row each; also the resume checkpoint |
| `average_inversions_plot.pdf` | inversions against round number |
| `signal_<tag>_cumulative_proportions.pdf` | cumulative proportion of each selection signal's effect on candidate ordering |
