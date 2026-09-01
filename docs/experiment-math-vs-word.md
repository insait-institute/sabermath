# Math vs. words

For each target, compares how similar its **full statement**, its **equation-only** content and its **word-only** content are to its top candidates.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs (see [experiment-evaluation.md](experiment-evaluation.md)).
- `scripts/config/math_vs_word.yaml` names the input and output Hugging Face datasets. Every step takes `--config-file` to point at a different one.
- Credentials as token files at the repo root: `.hftok`, `.geminitok`, `.openroutertok`. An exported `GEMINI_API_KEY` takes precedence over `.geminitok`.
- `HF_TOKEN` set for writing on the given repository.
- `--method`: one of the models, as given by `load_models.py`

## Usage
```bash
# 1. standardize LaTeX across targets and candidates
python scripts/analysis/fix_latex.py

# 2. split each problem into equation-only and word-only content
python scripts/analysis/strip_words_and_math.py

# 3. compute similarities, one method per invocation
python scripts/analysis/math_vs_word.py --method "Qwen/Qwen3-Embedding-8B"
python scripts/analysis/math_vs_word.py --method bm25
python scripts/analysis/math_vs_word.py --method <METHOD> --force-recalc

# 4. the figure
python scripts/plots/plot_math_vs_word.py          # selected models
python scripts/plots/plot_math_vs_word.py --all    # every method

# 5. the table
python scripts/report_experiments.py math-vs-word
python scripts/tables/math_vs_word_table.py --overall-only   # no HF access needed
```

Long runs can be split with `--shards N --shard I` and reassembled with `python scripts/analysis/math_vs_word_merge.py`.

### Instructions

```bash
python scripts/analysis/math_vs_word.py --method <MODEL_ID> --instruction p1
python scripts/analysis/math_vs_word_coverage.py --instructions p0 p1 p2 p3 --emit-commands
```

You can skip the `p0` instruction, if you already have the main results, as those are the same.

## Output

```
results/math_vs_word/similarities/<method>[__<instruction>].json
results/math_vs_word/similarities_baseline/<method>.json
results/math_vs_word/plots/*.pdf
results/math_vs_word/results_table.csv
results/tables/RESULTS_math_vs_word.md
results/tables/math_vs_word_instructions.tex
```