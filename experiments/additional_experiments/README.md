# Experiments

The following scripts are available for running the different experiments.

---

## 1. Run prompts for all candidate pairs for selected targets

Runs prompts for every pair of candidates associated with a specified list of target indices from the targets dataset.

### Example

```bash
python all_prompts_run.py \
  --config config.yaml \
  --mode no-ties \
  -n 1 \
  --indices 0 1 2 \
  --max-retries 3 \
  --overwrite
```

### Arguments

- `--config` — Path to the configuration file
- `--mode` — Evaluation mode (e.g. `no-ties`)
- `-n` — Number of runs per comparison
- `--indices` — Target indices to evaluate
- `--max-retries` — Maximum retry attempts for failed API calls
- `--overwrite` — Overwrite existing outputs

---

## 2. Run judge on specific `(target, candidate_a, candidate_b)` triples

Reads a CSV file containing explicit comparison triples and evaluates them using an LLM judge.

The script writes the judge outputs to the specified output file.

### Example

```bash
python all_specific_prompts_run.py \
  --config_file config.yaml \
  --pairs_csv pairs.csv \
  --mode no-ties \
  -n 1 \
  --max-retries 3 \
  --overwrite
```

### Arguments

- `--config_file` — Path to the configuration file
- `--pairs_csv` — CSV file containing `(target, candidate_a, candidate_b)` triples
- `--mode` — Evaluation mode
- `-n` — Number of runs per comparison
- `--max-retries` — Maximum retry attempts for failed API calls
- `--overwrite` — Overwrite existing outputs

---

## 3. Run ordinal ranking for selected targets

Queries an LLM to assign a score to every candidate for each specified target.

### Example

```bash
python ordinal_ranking.py \
  config.yaml \
  ordinal_ranking.csv \
  0 1 2 3 \
  output.csv
```

### Arguments

1. `config.yaml` — Configuration file
2. `ordinal_ranking.csv` — Input CSV containing candidates
3. `0 1 2 3` — Target indices to evaluate
4. `output.csv` — Output file for ranking results

---

## 4. Random-choice tournament experiment

Computes relevance scores for selected target indices using a Bradley–Terry model.

Unlike the Swiss tournament setup, candidates are paired randomly at each round.

### Example

```bash
python random_choice_rounds.py \
  --config_path config.yaml \
  --num_rounds 20 \
  --idxs 1 2 3
```

### Arguments

- `--config_path` — Path to the configuration file
- `--num_rounds` — Number of tournament rounds
- `--idxs` — Target indices to evaluate

### Notes

- The default value for `--num_rounds` is `20`.
- This matches the number of rounds used in the Swiss tournament experiments.

## 5. Plot number of inversions acorss number of rounds between Swiss-tour-derived scores and full tournament with Bradley-Terry

Simulates a real Swiss tournament with a number of rounds and calculates the number of inversions every 5 rounds. Then plots these numbers to give a visualization of how quickly the Swiss tournament ordering converges to the Full tournament's scores with increasing the number of rounds.

### Example

```bash
python avg_inversions.py \
    --config_file config.yaml \
    --output_json final_results.json \
    --matches_save_file matches_res_inv.csv \
    --num_rounds 50
```

Then:

```bash
python plot_inversions.py \
    --results_json final_results.json
```

## 6. Plot a cumulative proportion of the 3 different signals' effect on the ordering of the candidates

```bash
python signal_effect.py \
    --config_file config.yaml
```

Optional: you can add one of the following arguments:
--algebra
--geometry
--combinatorics
--calculus
--number_theory
To only plot results for the problems from this domain.