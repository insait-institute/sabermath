# Words vs. Equations Importance Experiments

This pipeline compares how embedding models represent the **full problem statement**, the **word-only content**, and the **equation-only content** of math problems relative to their top candidate problems/solutions.

Run the scripts in the order below.

> **Important:** Every `--model_id <MODEL_ID>` argument must be copied **exactly** from the supported model list below.

---

## 1. Standardize LaTeX formatting

```bash
python scripts/analysis/fix_latex.py
```

This reads the target and candidate datasets named in
`scripts/config/math_vs_word.yaml`, which is resolved next to
the module - pass `--config-file` only to use a different one.

It uses an LLM prompt to standardize the LaTeX formatting for:

- every target problem,
- every target solution,
- the problem and solution of the 150 candidates for each target.

The processed outputs are written to new Hugging Face datasets. Make sure the relevant input and output dataset names are included in `config.yaml`.

---

## 2. Extract equation-only and word-only content

```bash
python scripts/analysis/strip_words_and_math.py
```

This script performs regex-based extraction of:

- mathematical expressions,
- non-mathematical text.

It processes every target and every relevant candidate, then writes the extracted fields to the Hugging Face datasets specified in `config.yaml`.

---

## 3. Compute embedding similarities

```bash
python scripts/analysis/math_vs_word.py --method <METHOD>
```

The <METHOD> argument can be either a model id written exactly as in the list below, or one of 'jaccard', 'tf-idf', 'bm25' or 'approach0'. These are four non-embedding retrieval methods we also evaluate in our main experiment.

In the case of an embedding model method, each target, this script computes embeddings for:

- the full problem statement,
- the equation-only content,
- the word-only content.

It then compares each target representation against the embeddings of the target’s top 5 candidates, where each candidate is represented using its problem plus solution.

Similarly, for the non-embedding model methods (Jaccard similarity, TF-IDF, BM-25 and Approach0) the relevance of the target problem is computed to the 5 most relevant candidates. For TF-IDF all 150 candidates are used as the corpus. Tokenization in all 4 cases is done through Approach Zero's specialized mathematics tokenizer (equation blocks are tokenized via `pya0.tokenize`; the surrounding prose is tokenized separately as plain words, with no stop-word filtering).

The script averages the 5 similarity scores and saves the results as a JSON file in:

```bash
similarities/
```

The output filename is based on the embedding model name.

Example:

```bash
python scripts/analysis/math_vs_word.py --method "Qwen/Qwen3-Embedding-8B"
```

### How the models are called

Every method is built and scored by delegating to `scripts/run_experiments.py`:

```python
rr._build_experiment_processor(key, arm, tp, save_dir, legacy)
rr._experiment_scores_kwargs(key, instruction_text, legacy)
```

which is exactly what `scripts/run_dedup.py` does. The main experiment, the
dedup experiment and this one therefore call each model identically - same
backend, same pooling, same vendor input envelope, same chunking, same
processor arguments. `load_models.py`'s header block lists what this fixed
when it was introduced (2026-08-26) and why.

The default arm is **p0 under the canonical protocol**: no instruction text,
but the model's full vendor input envelope. It is not prompt-free.

`microsoft/codebert-base` is the one exception - it is not in the paper's
model table, has no registry key, and so keeps the generic
SentenceTransformers path.

> **Every `results/math_vs_word/similarities/` file produced before
> 2026-08-26 predates the vendor input envelopes**, and sits at the
> unsuffixed filename. Those numbers are not comparable to a run made after
> that change. `results/math_vs_word/similarities_baseline/` holds the
> snapshots taken before the recalculation.

### Running the sweep

One method per invocation, in an environment that suits it (the same four
families need their own - see
[experiment-evaluation.md](experiment-evaluation.md)):

```bash
python scripts/analysis/math_vs_word.py --method <METHOD>
python scripts/analysis/math_vs_word.py --method <METHOD> --force-recalc
```

Runs write into `results/math_vs_word/similarities/`, one file per method, so
methods are independent and a failure never blocks the rest. Long runs can be
split and reassembled with
`python scripts/analysis/math_vs_word_merge.py`.

Credentials are read from token files at the repo root: `.hftok`,
`.geminitok`, `.openroutertok` (all gitignored). An exported `GEMINI_API_KEY`
takes precedence over `.geminitok`, and `GOOGLE_API_KEY` is pinned to
whichever won, because google-genai gives `GOOGLE_API_KEY` precedence and a
stale one would otherwise slip through.

### Instruction ablation (optional)

```bash
python scripts/analysis/math_vs_word.py --method <MODEL_ID> --instruction p1
```

The instruction is wrapped onto the query with
`sabermath.instructions.format_instructed_query`, so an instructed
math-vs-word query is byte-identical to an instructed main-benchmark query.
All three target representations (full, equation-only, word-only) are
wrapped; the candidates are not.

Output goes to `similarities/<method>__<arm>.json`, so an ablation can never
overwrite the default file.

Two things to be careful about:

- **`p0` is the default arm.** `similarities/<method>.json` and
  `<method>__p0.json` are the same run, for every model.
  `math_vs_word_coverage.py --arms p0 --emit-commands` prints `cp` commands rather
  than run commands for exactly that reason.
- **Some models must NOT get the generic wrap.** ReasonIR takes the
  instruction as an `encode()` argument and masks those tokens out of its
  mean pool; the Qwen3-Reranker family takes it in the model's own
  `<Instruct>` slot at construction time. `load_models.wraps_instruction()`
  asks `sabermath.registry` which case applies - it is not guessed here.
- **The lexical methods have no instruction arm.** `jaccard`, `tf-idf`,
  `bm25` and `approach0` are the ablation's control rows and `--instruction`
  is refused for them.

`math_vs_word_coverage.py --arms p0 p1 p2 p3 --emit-commands` reports which arms are
missing and prints the exact command for each.

### The pre-2026-08-26 numbers

There is no longer a switch that reproduces them: the envelope-free protocol
and its `--legacy` flag were removed on 2026-08-31, here and in
`run_experiments.py`/`run_dedup.py` alike. The runs made under it are kept as
data in `results/math_vs_word/similarities_baseline/`, which is the only
remaining way to compare against them.

---

## Supported model IDs

The authoritative roster is
`sabermath.math_vs_word.load_models.ALLOWED_MODELS` (44 methods as of
2026-08-26: 40 models plus the four lexical controls). Run
`python scripts/analysis/math_vs_word_coverage.py` to print it along with which methods have a
complete result.

The 17 below were the original math-vs-word roster; the rest of the paper's
model table was added to `load_models.ADDITIONAL_MODELS` on 2026-08-22 and is
equally runnable. Use any of these strings **exactly** as the value of
`--method`:

```text
Qwen/Qwen3-Embedding-8B
Qwen/Qwen3-Embedding-4B
Qwen/Qwen3-Embedding-0.6B
BAAI/bge-m3
tencent/KaLM-Embedding-Gemma3-12B-2511
google/embeddinggemma-300m
google-bert/bert-base-uncased
FacebookAI/roberta-base
microsoft/codebert-base
google/gemini-embedding-001
google/gemini-embedding-2
microsoft/harrier-oss-v1-0.6b
microsoft/harrier-oss-v1-270m
microsoft/harrier-oss-v1-27b
Octen/Octen-Embedding-4B
Octen/Octen-Embedding-8B
jinaai/jina-embeddings-v5-text-nano
```

---

## 4. Plot ordering histograms

```bash
python scripts/plots/plot_math_vs_word.py --all   # --all is optional
```

In the case when an --all argument is passed, the plot is computed for all methods (embedding models and retrieval methods). In the general case only specific models and methods are plotted for visual clarity.

This reads all corresponding JSON files from:

```
results/math_vs_word/similarities/
```

For every method the scripts calculates how often the mathematical equations are more relevant to the target problem than the text-only content. These results are then plotted separately for every domain and completely across all domains. The plot is saved to the 'plots' folder.