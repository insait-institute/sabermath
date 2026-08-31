# Backend provenance

Why each model is served the way it is, and what that choice was checked
against. The registry (`src/sabermath/registry.py`) cites this file wherever a
recipe looks arbitrary but is not.

## The rule

Every model runs on ONE backend — its production backend — and a latency
measurement uses the same processor construction the nDCG run does. When a
model moved from SentenceTransformers to vLLM (the 2026-08-20 rollout), the
move was gated on reproducing a raw-HF reference, not assumed.

## Where the evidence lives

Every probe that produced those comparisons was removed on 2026-08-31: the
feasibility harness (`test_vllm_feasibility.py`), the HF-transformers
reference processors it compared against (`Rank1HFProcessor`,
`Qwen3RerankerProcessor`, `RaDeRRerankerProcessor`, and the hand-built
SentenceTransformers RaDeR bi-encoder stack), and the input-protocol probes
(`diagnose_protocol.py`, `repro_st_backend.py`, `repro_vllm_backend.py`,
`diag_tokenization.py`, the RaDeR arm sweeps). They answered questions that
are now settled; keeping them would mean maintaining a second scoring path
that nothing runs.

Their verdicts are archived as data:

    results/diagnostics/vllm_feasibility/summary.json     per-model verdicts
    results/diagnostics/vllm_feasibility/<model>__*.json  per-run detail
    results/diagnostics/vllm_feasibility_bf16/            the bf16 precision arm
    results/diagnostics/dtype_ab/                         rank1 fp16-vs-bf16
    results/diagnostics/protocol/                         input-envelope verdicts

Re-establishing any of these means writing a fresh comparison against a
raw-HF reference, which is the bar the original ones had to clear too.

A verdict records the Spearman correlation between the two backends'
candidate rankings and the resulting |ΔnDCG@10|. FEASIBLE meant Spearman
≥ 0.999 with |ΔnDCG@10| inside the bootstrap CI half-width.

## The findings worth keeping in mind

**Reduced precision was exonerated.** An early finding that bf16 "wrecks
rankings" (Spearman 0.19–0.39) was confounded: it was measured against a
chat-template-corrupted reference. The corruption was the reference's bug, not
dtype's. RaDeR now runs bf16, its training precision.

**Two models were being served WRONG on SentenceTransformers, and vLLM was
correct.** `llama-embed-nemotron-8b` and the Octen family both shifted their
math-vs-word statistic substantially when moved to vLLM, which looked like a
vLLM regression and was the opposite:

- `llama-embed-nemotron-8b` declares `LlamaBidirectionalModel` and becomes
  bidirectional through an override of `LlamaModel._update_causal_mask()`.
  transformers 5.x DELETED that method, so under an unpinned install the
  override is never called and the model runs CAUSAL while reporting nothing.
  Every ST number produced that way is a causal run of a bidirectional model.
- Octen's old loader HAND-BUILT a `[Transformer, Pooling(lasttoken),
  Normalize]` stack. The generic `SentenceTransformer` load agrees with vLLM to
  2e-04; the hand-built imitation agreed only to 0.70.

Both are the same shape of finding: an imitation of a vendor stack was the
outlier, not the engine. The rule that follows: load a model the way its
vendor documents, and treat any hand-built imitation of that stack as the
suspect when two backends disagree.

**Four families still need their own environment**, and will fail loudly in
the wrong one — see `scripts/envs/` and the docstring in
`scripts/run_experiments.py`:

| Family | Environment | Why |
|---|---|---|
| GTE / Reason-ModernColBERT | `env_colbert.yml` | pylate pins sentence-transformers 5.3.0 |
| SPLADE-code | `env_splade.yml` | SparseEncoder |
| INF-Retriever-v1-Pro, INF-X-Retriever | `env_inf_retriever.yml` | bidirectional remote code needs transformers 4.51.x, and has no validated vLLM recipe |
| ReasonIR-8B | `env_reasonir.yml` | transformers pinned to the version its remote code was written against |

**ReasonIR stays on SentenceTransformers on purpose.** vLLM cannot express its
instruction mechanism: its `encode()` runs a full bidirectional pass over
`instruction + text + embed_eos` and then zeroes the instruction positions in
the POOLING mask, which a stock MEAN pooler has no way to represent.
Prompt-free the two agree to cosine 0.9999, so `reasonir-8b-vllm` exists as a
p0-only key and hard-errors on instructed arms rather than returning numbers
from a mechanism that was not applied.

**rank1-32b is pinned to `tensor_parallel_size=1`.** Sharding it silently
corrupts its relevance scores — confirmed by direct diagnostic re-run. The pin
is enforced in the builder regardless of what `--tensor-parallel-size` is
passed. Data parallelism (independent single-GPU engines splitting the query
set) is the untried alternative.
