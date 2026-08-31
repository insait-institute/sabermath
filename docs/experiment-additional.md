# Construction-pipeline analyses

Two checks on how the benchmark's relevance scores were produced. Both read
the tournament configuration from
`src/sabermath/analysis/config_additional.yaml`, resolved next to the module,
so `--config_file` is only needed to point at a different one.

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
python -m sabermath.analysis.inversions \
    --output_json final_results.json \
    --matches_save_file matches_res_inv.csv \
    --num_rounds 50

python -m sabermath.analysis.plot_inversions --results_json final_results.json
```

`--num_rounds 50` matches the rounds used in the Swiss tournament
experiments. The run checkpoints into `--matches_save_file` every
`--checkpoint_every` rounds and resumes from it, so an interrupted run does
not restart; `--force_recalc` ignores an existing checkpoint.

## Selection-signal effect

Candidates enter the benchmark through two signals — overlap in ontology
topics, and lexical overlap between solution-idea summaries. This plots the
cumulative proportion of each signal's effect on the final ordering of
candidates, separating candidates found by topic only, by solution summary
only, and by both.

```bash
python -m sabermath.analysis.signal_effect
```

Restrict it to one domain with any one of `--algebra`, `--geometry`,
`--combinatorics`, `--calculus`, `--number_theory`.
