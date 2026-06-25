# Words vs. Equations Importance Experiments

This pipeline compares how embedding models represent the **full problem statement**, the **word-only content**, and the **equation-only content** of math problems relative to their top candidate problems/solutions.

Run the scripts in the order below.

> **Important:** Every `--model_id <MODEL_ID>` argument must be copied **exactly** from the supported model list below.

---

## 1. Standardize LaTeX formatting

```bash
python fix_latex.py --config_file config.yaml
```

This script reads the target and candidate datasets specified in `config.yaml`.

It uses an LLM prompt to standardize the LaTeX formatting for:

- every target problem,
- every target solution,
- the problem and solution of the 150 candidates for each target.

The processed outputs are written to new Hugging Face datasets. Make sure the relevant input and output dataset names are included in `config.yaml`.

---

## 2. Extract equation-only and word-only content

```bash
python strip_words_and_math.py --config_file config.yaml
```

This script performs regex-based extraction of:

- mathematical expressions,
- non-mathematical text.

It processes every target and every relevant candidate, then writes the extracted fields to the Hugging Face datasets specified in `config.yaml`.

---

## 3. Compute embedding similarities

```bash
python calc_sims.py --method <METHOD> --config_file config.yaml
```

The <METHOD> argument can be either a model id written exactly as in the list below, or one of 'jaccard','tf-idf' or 'approach0'. These are three non-embedding retrieval methods we also evaluate in our main experiment.

In the case of an embedding model method, each target, this script computes embeddings for:

- the full problem statement,
- the equation-only content,
- the word-only content.

It then compares each target representation against the embeddings of the target’s top 5 candidates, where each candidate is represented using its problem plus solution.

Similarly, for the non-embedding model methods (Jaccard similarity, TF-IDF and Approach0) the relevance of the target problem is computed to the 5 most relevant candidates. For TF-IDF all 150 candidates are used as the corpus. Tokenization in all 3 cases is done through Approach Zero's specialized mathematics tokenizer.

The script averages the 5 similarity scores and saves the results as a JSON file in:

```bash
similarities/
```

The output filename is based on the embedding model name.

Example:

```bash
python calc_embedding_sims.py --method "Qwen/Qwen3-Embedding-8B" --config_file config.yaml
```

---

## Supported model IDs

Use one of the following strings **exactly** as the value of `--method`:

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
python plot_hist.py --config_file config.yaml --all (optional)
```

In the case when an --all argument is passed, the plot is computed for all methods (embedding models and retrieval methods). In the general case only specific models and methods are plotted for visual clarity.

This script reads all corresponding JSON files from:

```bash
similarities/
```

For every method the scripts calculates how often the mathematical equations are more relevant to the target problem than the text-only content. These results are then plotted separately for every domain and completely across all domains. The plot is saved to the 'plots' folder.