# Construction of Informal Math Embedding Model Benchmark

The construction of the benchmark starts with the following files:

```
<HF_DATABANK> - A HuggingFace dataset of ~283k problems with `problem` and `solution` columns
data/tree.json - A JSON tree of Mathematical Tags
```

Note: Currently the tree.json file is stored at `assets/tree.json`. You can move that to a `data/tree.json`.

Due to the large scale of the data bank the pipeline consists of two phases:

- Phase 1: Using fast metrics to compute the pairwise similarity of all ~80B pairs of problems and using those to select good target problems and a list of candidate problem for reranking for each of the targets

- Phase 2: Assigning relevance scores for each selected target-candidate pair using a Bradley-Terry model and an LLM judge, then scaling the final scores by 5.

Run all commands below from the repository root unless the command explicitly changes directory.

## Setup

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate build-benchmark
pip install -e ./fastbma
```

Alternatively, install the Python requirements directly into an existing Python 3.11 environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If using the official OpenAI API for annotation, set:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

Set the paths used below:

```bash
export HF_DATABANK=<input_hf_path>
export HF_TAGS=<tags_output_hf_path>
export HF_IDEAS=<ideas_output_hf_path>
export HF_ANNOTATED=<annotated_output_hf_path>
export HF_SELECTION_PREFIX=<selection_output_hf_path_prefix>
export HF_RELEVANCES=<raw_relevance_output_hf_path>
export HF_FINAL=<final_output_hf_path>
mkdir -p data/sim
```

## Phase 1: Selecting targets and candidates

Filtering is done based on two metrics. The first is BMA (Best-Match Average) on lists of tags assigned on each problem using a small LLM-based framework. Tags themselves are chosen from the nodes of `tree.json`. The second metric is Jaccard similarity on the "core ideas" of problem - those are one sentence descriptions of the main idea used to solve the problem. Those are also annotated by an LLM. To select "good" targets we filter our only problems with at least 50 candidates with "high" BMA and "low" Jaccard, at least 50 with "low" BMA and "high" Jaccard and at least 50 with both "high" BMA and Jaccard. Here "high" for both BMA and Jaccard means scores above certain threshold - selected as `BMA_THRESHOLD=0.8579101562504547` and `JACCARD_THRESHOLD=0.21057128906295475`, and "low" means all other scores.

### Step 1: Annotate
In this step tags and core ideas are generated via the scripts at `./annotate`. For the exact annotation commands and options, follow the inner README files:

```
annotate/tags/README.md
annotate/ideas/README.md
annotate/postprocess/README.md
```

1. Assign a list of tags to each problem in the dataset. See `./annotate/tags` (The main script is `make_tags.py`)

2. Extract only the core idea of each problem (20-30 words). See `./annotate/ideas` (The main script is `extract_ideas.py`)

3. Postprocess the dataset - combine results from 1. and 2.; clear up tags and ideas. Check `./annotate/postprocess/post_process_datasets.py`

4. Postprocess the tag tree - extract frequencies of each tag in the anotated dataset; drop those never assigned; add IDs to nodes. Check `./annotate/postprocess/extract_new_tree.py`

The resulting files of this step are:

```
<HF_ANNOTATED> - A ~283k problem dataset of problems with their tag lists and core ideas extracted
data/freq_tree.json - The tag tree with frequencies and ids assigned to each node
```

### Step 2: Compute pairwise similarities (BMA and Jaccard)
Done via the scripts at `./similarities`.

1. Compute pairwise Lin Similarities of each two nodes of `freq_tree.json` and export as a similarity matrix. This is done to avoid redundant computations during the next step. Check `./similarities/node_similarities`. This outputs a `tree_lin_sim.npy` file used in 2.

```bash
python similarities/node_similarities/node_similarities.py \
  data/freq_tree.json \
  data/tree_lin_sim.npy
```

2. Compute both BMA and Jaccard similarities. Check `./similarities/compute`

```bash
python similarities/compute/compute_similarities.py "$HF_ANNOTATED" \
  --tree data/freq_tree.json \
  --similarities data/tree_lin_sim.npy \
  --out data/sim \
  --chunk-count 100 \
  --max-parallel 16
```

Adjust `--max-parallel` to fit the machine. Omit it to use the script's automatic CPU selection, or use `--no-parallel` only for small/debug runs.

The output of this step is the directory:

```
data/sim/ - Contains files chunk_0.bin, chunk_1.bin, ... that store the BMA and Jaccard scores
```

### Step 3: Select good targets and their candidate lists
The final step of the filtering: it is done using the script `./select/select_targets_and_candidates.py` using the outputed thresholds `BMA_THRESHOLD=0.8579101562504547` and `JACCARD_THRESHOLD=0.21057128906295475`. The outputs are two HuggingFace datasets - the first contains the targets and the second - the candidates. The target dataset has a `candidates` column: each row of it contains a list of indices each pointing out to a candidate in the candidates dataset by the number of the row.

```bash
python select/select_targets_and_candidates.py "$HF_ANNOTATED" \
  --bma "$BMA_THRESHOLD" \
  --jaccard "$JACCARD_THRESHOLD" \
  --per-group-count 50 \
  --similarities data/sim \
  --out "$HF_SELECTION_PREFIX"
```

This creates:

```
${HF_SELECTION_PREFIX}_targets
${HF_SELECTION_PREFIX}_candidates
```

### Step 5: Sample the final target set
The final sampler keeps the hard-coded domain distribution in `select/final_sampler.py`, removes unused candidates, and remaps the candidate indices.

```bash
python select/final_sampler.py \
  "${HF_SELECTION_PREFIX}_targets" \
  "${HF_SELECTION_PREFIX}_candidates" \
  --seed 0
```

This creates:

```
${HF_SELECTION_PREFIX}_targets_reduced
${HF_SELECTION_PREFIX}_candidates_reduced
```

## Phase 2: Assigning relevance scores

Phase 2 is run from `./generate_scores`. For the exact scoring commands, configuration format, resume behavior, and Bradley-Terry options, follow the inner README:

```
generate_scores/README.md
```

The expected input of this phase is the reduced targets and candidates produced in Phase 1:

```
${HF_SELECTION_PREFIX}_targets_reduced
${HF_SELECTION_PREFIX}_candidates_reduced
```

The raw output of this phase is a HuggingFace dataset containing the target rows plus a `relevance_scores` column with one normalized Bradley-Terry score for each of the 150 candidates in the target row's `candidates` list.

Set the raw and final output paths before running the final transform:

```bash
export HF_RELEVANCES=<raw_relevance_output_hf_path>
export HF_FINAL=<final_output_hf_path>
```

Finally, run `final_transform.py` from the repository root to multiply all `relevance_scores` by 5:

```bash
python final_transform.py "$HF_RELEVANCES" --out "$HF_FINAL"
```

The final benchmark dataset is pushed to:

```
<HF_FINAL>
```