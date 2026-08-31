# Input protocol: what each model actually receives

Every model in this benchmark is fed through one of two protocols.

- **Current (default).** Each model gets the input envelope its vendor
  documents, and every arm of the instruction ablation differs from its
  baseline in the prompt text and nothing else.
- **Legacy (`--legacy`).** Byte-for-byte reproduction of the runs made before
  2026-08-25, when several models were sent bare text their vendors do not
  define. Kept because the published paper/rebuttal numbers were produced
  that way, and because the two protocols together give a free
  "prompt-template sensitivity" comparison across the whole roster.

Legacy results are written with a `__legacy` tag in the filename and in their
own checkpoint directory, so runs can share `results/evaluation/` without
ever overwriting each other, and each run records the protocol that produced
it.

To see what a model actually receives, read its entry in
`src/sabermath/registry.py`: the envelope is an ordinary part of the model's
spec dict, and the table below is that file in prose.

---

## Where the envelope lives

Per-model prompts are ordinary entries in the model spec dicts in
`scripts/run_experiments.py`, next to the pooler config and the chunking
protocol — `query_prompt`, `document_prompt`, `query_suffix`,
`document_suffix`, `query_encode_kwargs`, `document_encode_kwargs`.

`EmbeddingProcessor.get_scores` applies them per side, **before** the vector
cache, so an affixed text is simply a different cache key — there is no
stale-cache hazard and no second engine round trip. The one exception is a
per-side *API parameter* (Gemini's `task_type`), which cannot be expressed as
text: those models split into a document call and a query call, and their
cache keys carry the side tag so a text embedded as a query is never served
from a document embedding.

`scripts/models.txt` carries the same strings on the `--encode-kwargs` lines
for the production/paper runs, and `scripts/run_timing.py` imports
them straight from `run_experiments.py` so the timing harness can never measure
a configuration production no longer uses.

## Per-model envelope

| Model | Query | Document | Source |
|---|---|---|---|
| `multilingual-e5-large` | `query: ` | `passage: ` | model card, **required** ("otherwise you will see a performance degradation") — the repo ships no `config_sentence_transformers.json`, so this pair is card-sourced and not machine-checkable |
| `embeddinggemma-300m` | `task: search result \| query: ` | `title: none \| text: ` | packaged prompt grammar |
| `jina-embeddings-v5-text-nano` / `-small` | `Query: ` | `Document: ` | `config_sentence_transformers.json` prompts |
| `rader-3b` / `-7b` / `-14b` | `query: {q}<\|im_end\|>` | `document: {d}<\|im_end\|>` | their `retrievers.py`, last-token pooling |
| `gemini-embedding-001` | `task_type=RETRIEVAL_QUERY` | `task_type=RETRIEVAL_DOCUMENT` | API enum (live-probed; see below) |

Models not listed get no envelope, in either protocol: their vendor either
documents none (`bge-m3`, `bert-base-uncased`, `roberta-base`, `splade-code-*`,
`text-embedding-3-*`, the lexical baselines) or documents exactly the
free-text `Instruct:`/`Query:` mechanism the ablation already uses
(Qwen3-Embedding and its relatives, Octen, Diver, Harrier, Nemotron,
Reason-Embed, KaLM, INF-Retriever).

### Notes on individual entries

- **jina-v5.** Both repos ship
  `prompts {"query": "Query: ", "document": "Document: "}` with
  `default_prompt_name: "document"` — read straight off the checkpoints on
  2026-08-25 (probe archived at
  `results/diagnostics/protocol/jina_verdict.json`). The two backends were
  doing different wrong things:
  - **nano (SentenceTransformers)**: a prompt-less `encode()` call is
    **bit-identical** to `prompt_name="document"` (cosine 1.000000), so
    queries really were embedded with the document prompt. The `query` and
    `document` prompts differ at cosine 0.935, and dropping the prompt
    entirely moves the vector to cosine 0.56 — this model is strongly
    prompt-sensitive, so "no prompt" is not a neutral choice either. The spec
    therefore also clears `default_prompt_name` (`disable_st_default_prompt`;
    `--clear-st-default-prompt`) so the explicit per-side
    prompts are not stacked on the repo default.
  - **small (vLLM)**: vLLM does not read that config and adds nothing of its
    own — its raw-text output matches the `prompt=""` reference at cosine
    0.9999. So this model was getting **no prompt at all**, and the explicit
    prefixes are required and do not double up. It stays on vLLM.
- **Gemini.** Live-probed against the real API on 2026-08-25
  (probe archived at
  `results/diagnostics/protocol/gemini.json`). On **gemini-embedding-001** the
  parameter is real: `RETRIEVAL_QUERY` returns byte-identical vectors to
  sending no config at all — it is the server-side default — while
  `RETRIEVAL_DOCUMENT` differs at cosine 0.80–0.88 on benchmark-like text. So
  only the DOCUMENT side of that model actually changes. On
  **gemini-embedding-2** all eight valid task types return byte-identical
  vectors while an invalid one still 400s: the parameter is accepted and
  validated but has no effect. `-2` is therefore deliberately left with no
  envelope — sending an inert parameter would buy nothing and would split one
  embed call into two. Re-probe before assuming this is still true.
- **RaDeR** suffixes are the tokenizer's EOS, verified against all three
  checkpoints rather than trusted from a constant.

  **The suffix used to be silently deleted.** The envelope is applied to the
  text and the text is then chunked, and `_chunk_texts` round-trips through
  `tokenizer.decode(..., skip_special_tokens=True)` — which dropped
  `<|im_end|>` from *every* RaDeR text, single-chunk ones included (found
  2026-08-25 by a 10x slowdown in a smoke run, then reproduced directly). For
  a **last-token pooler** that is exactly the position being pooled, so the
  half of the envelope that matters most never reached the model. The chunker
  and the truncator now decode with `skip_special_tokens=False`: the encode
  side passes `add_special_tokens=False`, so the only special tokens present
  are ones the caller deliberately put in the text. Verified not to introduce
  stray specials for the other chunked models (bert / roberta / e5).
  Both halves were asserted before the change landed — that
  `tokenizer.eos_token` matches the configured suffix, and that the suffix
  survives chunking on a single-chunk and a multi-chunk sample; the strings
  read off each checkpoint are archived at
  `results/diagnostics/protocol/vendor_strings.json`.
- **ReasonIR gets no envelope, and that is the faithful choice.** Measured
  against the official remote-code encoder on 2026-08-25
  (probe archived at
  `results/diagnostics/protocol/reasonir_verdict.json`). Its `encode()` builds
  `instruction + text + embed_eos`, runs one full bidirectional pass, and only
  then zeroes the instruction positions in the pooling mask. With
  `instruction=""` — the document side, and the whole prompt-free protocol —
  it prepends **nothing**. So bare text through the production vLLM mean
  pooler reproduces the official vectors at cosine **0.9999** (Spearman 1.0 on
  candidate ranking), while adding an `<|embed|>\n` wrapper *drops* that to
  **0.879**, because the wrapper tokens land inside vLLM's mean where the
  official code has them masked out. An earlier version of this document had
  the wrapper as the canonical form; the measurement says otherwise.

  For the *instructed* arms the official mechanism is "prepend
  `<|user|>\n{task}\n<|embed|>\n`, then exclude exactly those tokens from the
  pool". vLLM cannot exclude them: both vLLM approximations (the `<|user|>`
  wrapper, or plain `Instruct:`/`Query:` text) sit at cosine ~0.80 and
  Spearman 0.93 against the official encoder. The exact route is
  `ReasonIRProcessor` with per-side `instruction=` encode kwargs, which
  `EmbeddingProcessor.get_scores` now supports — see
  `REASONIR_INSTRUCTION_NOTE` in `run_experiments.py`. Decide that when the
  instruction arms are run; the prompt-free protocol needs nothing.

## Instruction handling

`src/sabermath/instructions.py`:

| Template | String |
|---|---|
| `canonical` (default) | `Instruct: {instruction}\nQuery: {query}` |
| `legacy` | `Instruct: {instruction}\n\nQuery: {query}` |

Every vendor card that documents this mechanism uses a **single** newline; the
double newline was inherited from the co-author's original script. Pick the
axis explicitly with `--instruction-template` (files then get an `nl1`/`nl2`
tag) to measure whitespace sensitivity on its own.

**Placement.** The instruction now wraps the *finished* query, after the task
transform:

```
Instruct: {i}\nQuery: Problem: {stmt}\n\nSolution: {sol}      (current)
Problem: Instruct: {i}\n\nQuery: {stmt}\n\nSolution: {sol}    (legacy)
```

Legacy mapped the instruction into the `problem` column *before* the
transform, so on `full-full` the instruction read as part of a problem
statement. `statement-statement` and `statement-full` are identical under both
placements — asserted end to end before the change landed, so only
`full-full` numbers move.

**Prompt keys.** `p0` no instruction, `p1`/`p2` the co-author's originals, `p3`
the rebuttal's short instruction, `pm` the repo's own production math
instruction (`Qwen3RerankerProcessor.DEFAULT_TASK_INSTRUCTION`, byte-identical
— asserted before the change landed).

## Arms that must differ in one variable only

- **Qwen3-Reranker** has a genuine `<Instruct>` slot. Its legacy `p0` filled
  that slot with the repo's own math instruction, so those rows compared *our
  instruction vs. their instruction* while every other row compared *nothing
  vs. instruction*. `p0` now uses the vendor's own default
  (`Given a web search query, retrieve relevant passages that answer the
  query`), and the production configuration lives on as `pm`.
  `DEFAULT_TASK_INSTRUCTION` itself is unchanged: it produced the published
  Qwen3-Reranker numbers and is a legitimate production choice — it just has
  to be disclosed in the paper as an *instructed* configuration rather than a
  bare baseline.
- **ColBERT** pads queries to `query_length` with mask tokens (query
  augmentation), so the query representation changes even for identical text.
  Legacy applied `query_length=256` only to instructed arms, against
  checkpoint defaults of 48 (GTE) and 128 (Reason). It is now 256 in every
  arm. Note for the paper: GTE's 48-token budget truncates most SABER-Math
  queries, so its published score is measured on heavily truncated queries.
- **Diver-GroupRank** has the same problem with `scaffold_reserve_tokens`
  (1024 baseline vs 1280 instructed changes the per-document token budget);
  it is now 1280 in every arm.

## Controls

Models with no vendor-documented free-text instruction mechanism are listed in
`INSTRUCTION_CONTROL_REASONS` (`scripts/run_experiments.py`) and reported in a
separate block by `scripts/report_experiments.py`. Their `p1`–`p3` rows
are not instruction-following measurements — but they are worth running: a
flat control block is exactly the evidence that movement on the instructable
block is real. This is deliberately *not* `INSTRUCTION_EXCLUDED`, which
hard-errors for the three models where instructed text is actively harmful
(`tf-idf`, `jaccard`, `approach0`).

## Known limitation, not fixed

Chunked models (`roberta-base`, `bert-base-uncased`, `multilingual-e5-large`
at 512 tokens; the RaDeR family at 2048) split long texts into chunks and mean
the chunk vectors. The instruction lands only in chunk 1 and is diluted by the
average. Left as a stated limitation.

---

## Where this lives in the code

The envelope is not a separate registry: it is part of each model's ordinary
spec dict, so there is one place to look per model and no second source of
truth to drift from.

| What | Where |
|---|---|
| Every model key, its processor recipe and its input envelope | `src/sabermath/registry.py` |
| Prompt keys, the `Instruct:`/`Query:` template, `format_instructed_query` | `src/sabermath/instructions.py` |
| Per-side affixes applied to the text, before the cache | `src/sabermath/processors/embedding_processor.py` |
| Two-call path for per-side API parameters, with side-tagged cache keys | `src/sabermath/processors/embedding_processor.py` |
| `task_type` threaded through to `embed_content(config=...)` | `src/sabermath/processors/google_processor.py` |
| Instruction applied to the finished query | `src/sabermath/benchmark.py` |
| Running one (model, prompt) cell, checkpointing, shard selection | `src/sabermath/runner.py` |
| Stitching shards back together | `src/sabermath/shards.py` |
| The timing harness imports the envelope from the registry, so it cannot drift | `scripts/run_timing.py` |

There is no protocol switch. The envelope-free path and its `--legacy` flag
were removed on 2026-08-31: every run this repo can now produce applies the
vendor envelopes described above.
