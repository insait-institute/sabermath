```markdown
# Computing Relevance Scores with Swiss Tournament Judging

As described in our paper, after sampling 150 candidate documents for each target query, we compute a relevance ordering over those candidates.

The pipeline has two stages:

1. **Simulate a Swiss tournament** over the 150 candidates using an LLM as a pairwise judge.
2. **Convert pairwise match outcomes into relevance scores** using the Bradley-Terry model.

The final output is a Hugging Face dataset containing the original targets plus computed relevance scores.

---

## 1. Run the Swiss Tournament

```bash
python swiss_tour_run.py --config_file config.yaml
```

This script compares candidate documents pairwise for each target query. In each match, the LLM is shown:

- the target problem and solution,
- two candidate problems and solutions,
- a prompt asking which candidate is more relevant.

The LLM must return either `\boxed{1}` or `\boxed{2}`. The winning candidate receives a Swiss tournament point.

By default, the script runs the number of rounds specified in the config file. In our experiments, we use 20 rounds.

### Output

The Swiss tournament results are saved to the CSV file specified by:

```yaml
swiss_tournament:
  matches_save_file: "matches_results_swiss.csv"
```

The output CSV has the following columns:

| Column | Description |
| --- | --- |
| `target_id` | ID of the target query/problem |
| `model_a` | ID of the first candidate document |
| `model_b` | ID of the second candidate document |
| `winner` | `1` if `model_a` won, `2` if `model_b` won |

---

## 2. Compute Bradley-Terry Relevance Scores

After the Swiss tournament matches have been generated, run:

```bash
python bradley_terry_run.py --config_file config.yaml --mode SWISS
```

This script reads the pairwise match results and fits a Bradley-Terry model for each target query. The resulting scores are normalized to the range from `0` to `1`.

The normalized scores are stored in the target examples under:

```python
relevance_scores
```

The resulting dataset is pushed to the Hugging Face Hub dataset specified by:

```yaml
hf_datasets:
  targets_with_relevances_dataset:
```

---

## Computing Scores for Selected Target Indices

By default, `bradley_terry_run.py` computes scores for all targets.

To compute scores only for selected target indices, pass `--idxs`:

```bash
python bradley_terry_run.py --config_file config.yaml --mode SWISS --idxs 0 1 2 3
```

---

## Modes

`bradley_terry_run.py` supports two modes:

| Mode | Description |
| --- | --- |
| `SWISS` | Reads the Swiss tournament matches from the CSV file specified in `config.yaml`. |
| `FULL` | Reads exhaustive pairwise comparison files from `calculated_csvs/matches_all_{i}.csv`, where `i` is the target index. |

Example using full pairwise comparisons:

```bash
python bradley_terry_run.py --config_file config.yaml --mode FULL --idxs 0
```

In `FULL` mode, the normalized scores are stored under:

```python
relevance_scores_full
```

---

<details>
<summary><strong>Configuration file</strong></summary>

Example `config.yaml`:

```yaml
hf_datasets:
  original_targets_dataset: "chosen_targets"
  original_candidates_dataset: "filtered_candidates"
  targets_with_relevances_dataset: "targets_relevances_computed"

prompts_pairwise_comparison:
  conf-score: "prompts/prompt_conf_score.txt"
  no-ties: "prompts/prompt_no_ties.txt"
  ties-allowed: "prompts/prompt_with_ties.txt"

swiss_tournament:
  num_rounds: 20
  matches_save_file: "matches_results_swiss.csv"
```

### Required fields

| Config key | Description |
| --- | --- |
| `hf_datasets.original_targets_dataset` | Hugging Face dataset containing the target problems. Each target must contain a list of candidate indices. |
| `hf_datasets.original_candidates_dataset` | Hugging Face dataset containing the candidate documents/problems. |
| `hf_datasets.targets_with_relevances_dataset` | Hugging Face dataset where the scored targets will be pushed. |
| `prompts_pairwise_comparison.no-ties` | Prompt template used for pairwise judging without ties. |
| `swiss_tournament.num_rounds` | Number of Swiss tournament rounds to run. |
| `swiss_tournament.matches_save_file` | Path to the CSV file where pairwise match results are saved. |

</details>

---

<details>
<summary><strong>Expected dataset format</strong></summary>

The target dataset is expected to contain examples with at least the following fields:

| Field | Description |
| --- | --- |
| `id` | Unique target ID |
| `problem` | Target problem text |
| `solution` | Target solution text |
| `candidates` | List of indices into the candidate dataset |

The candidate dataset is expected to contain examples with at least:

| Field | Description |
| --- | --- |
| `id` | Unique candidate ID |
| `problem` | Candidate problem text |
| `solution` | Candidate solution text |

Each target should have 150 candidate indices in its `candidates` field.

</details>

---

## Full Pipeline

Run the complete pipeline with:

```bash
python swiss_tour_run.py --config_file config.yaml
python bradley_terry_run.py --config_file config.yaml --mode SWISS
```

After completion, the output dataset will contain target examples with a `relevance_scores` field containing one normalized relevance score for each of the 150 candidates.
```