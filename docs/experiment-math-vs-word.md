# Math vs. words

For each target, compares how similar its **full statement**, its
**equation-only** content and its **word-only** content are to its top
candidates — then reports how often the equations win.

## Prerequisites

- `python -m pip install -e .` and the environment that model family needs
  (see [experiment-evaluation.md](experiment-evaluation.md)).
- `scripts/config/math_vs_word.yaml` names the input and output Hugging Face
  datasets. Every step takes `--config-file` to point at a different one.
- Credentials as token files at the repo root: `.hftok`, `.geminitok`,
  `.openroutertok`. An exported `GEMINI_API_KEY` takes precedence over
  `.geminitok`.
- Steps 1 and 2 write new Hugging Face datasets, so `HF_TOKEN` must be set and
  have write access.
- `--method` takes a model id exactly as listed by
  `sabermath.math_vs_word.load_models.ALLOWED_MODELS`, or one of `jaccard`,
  `tf-idf`, `bm25`, `approach0`.

## Usage

Run the steps in order. 1 and 2 are one-off preparation.

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

Step 3 scores each target against its top 5 candidates, each candidate
represented as problem plus solution, and averages the 5 scores. TF-IDF fits
its corpus on all 150 candidates. Lexical tokenization goes through Approach
Zero: equation blocks via `pya0.tokenize`, the surrounding prose as plain
words with no stop-word filtering.

Long runs can be split with `--shards N --shard I` and reassembled with
`python scripts/analysis/math_vs_word_merge.py`.

### Instructions

```bash
python scripts/analysis/math_vs_word.py --method <MODEL_ID> --instruction p1
python scripts/analysis/math_vs_word_coverage.py --instructions p0 p1 p2 p3 --emit-commands
```

`p0` **is** the default instruction: `<method>.json` and `<method>__p0.json` are the
same run. `--instruction` is refused for the four lexical methods, which are
the control rows. `math_vs_word_coverage.py` reports which instructions are missing
and prints the command for each.

## Output

```
results/math_vs_word/similarities/<method>[__<instruction>].json
results/math_vs_word/similarities_baseline/<method>.json   # pre-envelope runs, kept for comparison
results/math_vs_word/plots/*.pdf
results/math_vs_word/results_table.csv
results/tables/RESULTS_math_vs_word.md
results/tables/math_vs_word_instructions.tex
```

Each similarity file is keyed by target id, holding that target's three mean
similarities: `pr_full_vs_candidates`, `pr_math_vs_candidates`,
`pr_text_vs_candidates`.

The figure gives, per domain and overall, how often equations beat words. The
table gives one row per method and one column per instruction (`No Instr.`, `I1`,
`I2`, `I3`), best baseline first, with the lexical methods under "Instruction
controls". `--overall-only` skips the per-domain CSV columns, which are the
only part needing the targets dataset.
