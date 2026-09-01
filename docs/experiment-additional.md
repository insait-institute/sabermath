# Construction-pipeline checks

Two checks on how the benchmark's relevance scores were produced.

## Prerequisites

- `python -m pip install -e ".[analysis,lexical]"` — `[analysis]` provides `pandas`, `matplotlib` and `seaborn`, `[lexical]` provides `scikit-learn`.
- `scripts/config/tournament.yaml` names the datasets and the judging prompt. Every entry has a working public default; pass `--config-file` to use a different one.
- A judge for the tournament simulation. It defaults to a local vLLM model (`openai/gpt-oss-120b`), so it needs a GPU and no API key. Point `judge_api` in the config at a hosted provider to use one instead, and export that provider's key — `OPENAI_API_KEY`, `GOOGLE_API_KEY` or `OPENROUTER_API_KEY`.

## Usage

### Tournament convergence

Simulates a Swiss tournament and counts, every 5 rounds, the inversions between its ordering and a full round-robin scored with Bradley-Terry.

```bash
python scripts/analysis/inversions.py --num-rounds 50

python scripts/plots/plot_inversions.py \
    --results-json results/construction/final_results.json
```

The run checkpoints into `--matches-save-file` every `--checkpoint-every` rounds and resumes from it. Use `--force-recalc` to ignore an existing checkpoint.

### Selection-signal effect

```bash
python scripts/analysis/signal_effect.py
```

Restrict to one domain with `--algebra`, `--geometry`, `--combinatorics`, `--calculus` or `--number-theory`.

## Output

| Path | Contents |
|---|---|
| `results/construction/final_results.json` | `average_num_inversions_after_<n>_rounds` per checkpointed round; override with `--output-json` |
| `results/construction/matches_res_save.csv` | every pairwise match played, one row each; also the resume checkpoint. Override with `--matches-save-file` |
| `results/construction/average_inversions_plot.pdf` | inversions against round number |
| `results/construction/signal_<tag>_cumulative_proportions.pdf` | cumulative proportion of each selection signal's effect on candidate ordering |
