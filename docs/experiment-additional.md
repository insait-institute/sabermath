# Construction-pipeline analyses

Two checks on how the benchmark's relevance scores were produced. Both read
the tournament configuration from
`scripts/config/tournament.yaml`, so `--config-file` is only needed to point
at a different one.

The pipeline these analyse is in `build_benchmark/` — see
[`build_benchmark/README.md`](../build_benchmark/README.md) for the Swiss
tournament and Bradley-Terry scoring themselves.

## Tournament convergence: how many rounds are enough?

Simulates a Swiss tournament for a given number of rounds and counts, every 5
rounds, the inversions between its ordering and the ordering from a full
round-robin scored with Bradley-Terry. Plotting those counts shows how quickly
the reduced tournament converges on the full one — which is what justifies
running the reduced one.

```bash
python scripts/analysis/inversions.py \
    --output-json final_results.json \
    --matches-save-file matches_res_inv.csv \
    --num-rounds 50

python scripts/plots/plot_inversions.py --results-json final_results.json
```

`--num-rounds 50` matches the rounds used in the Swiss tournament
experiments. The run checkpoints into `--matches-save-file` every
`--checkpoint-every` rounds and resumes from it, so an interrupted run does
not restart; `--force-recalc` ignores an existing checkpoint.

## Selection-signal effect

Candidates enter the benchmark through two signals — overlap in ontology
topics, and lexical overlap between solution-idea summaries. This plots the
cumulative proportion of each signal's effect on the final ordering of
candidates, separating candidates found by topic only, by solution summary
only, and by both.

```bash
python scripts/analysis/signal_effect.py
```

Restrict it to one domain with any one of `--algebra`, `--geometry`,
`--combinatorics`, `--calculus`, `--number-theory`.
