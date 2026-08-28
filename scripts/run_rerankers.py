"""Evaluate the SABER-Math rerankers that need custom scoring logic (rank1
0.5B/7B/32B, Qwen3-Reranker 0.6B/4B/8B, GTE-ModernColBERT,
Reason-ModernColBERT, ReasonIR, SPLADE-code 0.6B/8B, RaDeR bi-encoders
3B/7B/14B, RaDeR-reranker-7B, Diver-GroupRank-32B, INF-Retriever-v1-Pro,
INF-X-Retriever), plus the models that run through the generic HuggingFace
path (Qwen3-Embedding-8B, Reason-Embed-Qwen3-8B, Diver-Retriever 0.6B/4B),
across all three tasks (statement-statement, statement-full, full-full).

Each model/task run is checkpointed after every query to
"<save-to>/.checkpoints/<model>/<subset>/<task>.json" (see
sabermath.benchmark.evaluate's checkpoint_dir). If a job is killed (e.g. a
SLURM wall-clock timeout - the exact failure mode that killed 6 of 7 past
rank1-32b attempts in rag-math-test/running-rerankers/), simply re-running
the identical command resumes from the last completed query instead of
starting over from zero.

Each model's result is saved as its own JSON file with the same
{"domains": [...], "reports": {<model>: {..., "ndcgs_by_task": {...}}}}
shape that testing/test_models.py produces, so
testing/compute_confidence_intervals.py can be pointed at the output
directory (or individual files) to get per-task, per-domain 95% CIs (normal
+ bootstrap) for free - no separate stats code needed for these models.

IMPORTANT - dependency isolation: since the 2026-08-20 vLLM-default rollout
most models here run from ONE environment, scripts/rerankers/envs/
env_vllm_feas.yml (vllm==0.26.0 pinned + peft for rader-reranker-7b's
one-off LoRA merge): rank1, diver-grouprank, every qwen3-reranker,
reasonir-8b, the rader bi-encoders + reranker, and all four GENERIC_MODELS.
Only three exceptions still need their own envs - GTE/Reason-ModernColBERT
(pylate pins sentence-transformers==5.3.0, env_colbert.yml), SPLADE
(SparseEncoder, env_splade.yml), and the inf-retriever-v1-pro /
inf-x-retriever pair (SentenceTransformers production path,
env_inf_retriever.yml - the retriever's bidirectional remote-code
attention has no VALIDATED vLLM recipe yet, see
_build_inf_retriever_processor, and that remote code needs the exact
transformers==4.51.x pin, see the env's header; the INF-X aligner is a
standard Qwen2.5 chat model happy in the same env).
env_st_biencoders.yml remains only for the LEGACY reference paths
(scripts/test_vllm_feasibility.py). Still run this script once per model
(or per same-env family), each in its own process:

    python scripts/run_rerankers.py --models rank1-32b
    python scripts/run_rerankers.py --models reasonir-8b
    python scripts/run_rerankers.py --models gte-moderncolbert reason-moderncolbert

Each invocation still runs its one (or few) models in an isolated subprocess
so a crash/OOM doesn't corrupt already-written results.

Usage:
    # One model, all 3 tasks
    python scripts/run_rerankers.py --models splade-code-8b

    # One model, one task
    python scripts/run_rerankers.py --models rank1-32b --task statement-full

    # Smoke test on a random 20-query subset before committing to a full,
    # possibly multi-hour run (rank1-32b in particular - see
    # running-rerankers/ in rag-math-test for why that matters: every past
    # attempt at a full 1000-query run there was killed by the SLURM time
    # limit before finishing).
    python scripts/run_rerankers.py --models rank1-32b --n 20
"""

import argparse
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sabermath import evaluate
from sabermath.data import load_data
from sabermath.instructions import (
    DEFAULT_INSTRUCTION_TEMPLATE,
    INSTRUCTION_TEMPLATES,
    INSTRUCTIONS,
)
from sabermath.processors.embedding_processor import AFFIX_KEYS
from sabermath.processors import (
    Approach0Processor,
    BM25Processor,
    ColBERTProcessor,
    EmbeddingProcessor,
    GoogleProcessor,
    GroupRankProcessor,
    INFXRetrieverProcessor,
    JaccardProcessor,
    OpenAIProcessor,
    OpenRouterEmbeddingProcessor,
    Qwen3RerankerProcessor,
    Qwen3RerankerVLLMProcessor,
    RaDeRRerankerProcessor,
    RaDeRRerankerVLLMProcessor,
    Rank1Processor,
    Rank1HFProcessor,
    ReasonIRProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    TfidfProcessor,
    VLLMProcessor,
)
from sabermath.schemas import Branch, Task

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

EXPERIMENT_COLBERT_QUERY_LENGTH = 256
EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE = 1280

# Qwen3-Reranker's own model-card default instruction. Used as the p0
# baseline for that family so p0 measures "no task instruction" like every
# other row does; DEFAULT_TASK_INSTRUCTION (the repo's production
# math-specific instruction) is available as prompt key "pm".
VENDOR_QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Models with no vendor-documented free-text instruction mechanism. Their
# p1-p3 rows are CONTROLS (a flat control row is the evidence that the effect
# on genuinely instructable models is real), not instruction-following
# measurements - aggregate_instructions.py reports them in a separate block.
# Deliberately NOT the same thing as INSTRUCTION_EXCLUDED, which hard-errors.
INSTRUCTION_CONTROL_REASONS = {
    "bge-m3": "vendor removed instruction prompting by design",
    "bert-base-uncased": "no instruction mechanism (plain MLM encoder)",
    "roberta-base": "no instruction mechanism (plain MLM encoder)",
    "splade-code-0.6b": "prompts {}; prompt_type picks a top-k budget, not text",
    "splade-code-8b": "prompts {}; prompt_type picks a top-k budget, not text",
    "bm25": "lexical, no instruction mechanism",
    "tf-idf": "lexical, no instruction mechanism",
    "jaccard": "lexical, no instruction mechanism",
    "bm25-no-tok": "lexical, no instruction mechanism",
    "tf-idf-no-tok": "lexical, no instruction mechanism",
    "jaccard-no-tok": "lexical, no instruction mechanism",
    "approach0": "structure search engine, no instruction mechanism",
    "embeddinggemma-300m": "fixed closed-set prompt grammar, free text is not the mechanism",
    "multilingual-e5-large": "fixed query:/passage: prefixes, free text is not the mechanism",
    "jina-embeddings-v5-text-nano": "fixed Query:/Document: prefixes + LoRA task adapter",
    "jina-embeddings-v5-text-small": "fixed Query:/Document: prefixes + LoRA task adapter",
    "gemini-embedding-001": "task_type enum API parameter, no free text",
    "gemini-embedding-2": "task_type enum API parameter, no free text",
    "text-embedding-3-small": "API exposes no instruction parameter at all",
    "text-embedding-3-large": "API exposes no instruction parameter at all",
    "gte-moderncolbert": "[Q]/[D] markers only, no task slot",
    "reason-moderncolbert": "[Q]/[D] markers only, no task slot",
    "rank1-0.5b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-7b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-32b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-32b-bf16": "no instruction slot; the vendor route is rewriting the query",
    "rank1-0.5b-hf": "no instruction slot; the vendor route is rewriting the query",
    "rank1-7b-hf": "no instruction slot; the vendor route is rewriting the query",
    "rank1-32b-hf": "no instruction slot; the vendor route is rewriting the query",
    "diver-grouprank-32b": "fixed rubric template, no task slot",
    "rader-reranker-7b": "T10 query:/Query:/document: template, no instruction slot",
}
INSTRUCTION_CONTROL_MODELS = frozenset(INSTRUCTION_CONTROL_REASONS)

# ReasonIR. Measured against the official remote-code encoder on 2026-08-25
# (scripts/instructions/diagnose_protocol.py reasonir, saved to
# results/protocol_diagnostics/reasonir_verdict.json). Its encode() does:
#
#     batch = [instruction + text + embed_eos for text in texts]
#     outputs = self(**inputs)                      # full bidirectional pass
#     if instruction: attention_mask[:, :len(instruction_tokens)] = 0
#     embeddings = self.pooling(last_hidden_state, attention_mask)
#
# so with instruction="" - the document side, and the whole prompt-free
# protocol - it prepends NOTHING. Sending bare text through the production
# vLLM mean pooler therefore reproduces the official vectors to cosine
# 0.9999 (Spearman 1.0 on candidate ranking), while adding an "<|embed|>\n"
# wrapper DROPS that agreement to 0.879: the wrapper tokens end up inside
# vLLM's mean, where the official code has them masked out. Prompt-free is
# the faithful configuration here, not a deviation from one.
#
# For the instructed arms the official mechanism is "prepend
# <|user|>\n{task}\n<|embed|>\n, then exclude exactly those tokens from the
# pool". vLLM cannot exclude them, and both vLLM approximations land at
# cosine ~0.80 / Spearman 0.93 against the official encoder. The exact route
# is ReasonIRProcessor with per-side encode kwargs, which
# EmbeddingProcessor.get_scores now supports directly:
REASONIR_QUERY_INSTRUCTION_SLOT = "<|user|>\n{instruction}\n<|embed|>\n"

# Tokenizer EOS expected by the RaDeR bi-encoder envelope below; checked at
# smoke time by scripts/instructions/verify_protocol.py.
RADER_EXPECTED_EOS = "<|im_end|>"

# Models needing a custom ModelProcessor (not resolvable from a bare HF name).
# rank1-32b's Processor CAN take tensor_parallel_size > 1 (vLLM generation,
# actually GPU-hungry at 32B), but a direct diagnostic re-run confirmed that
# doing so (tensor_parallel_size=2, matched to a 2-GPU allocation) silently
# corrupts its relevance scores - see the note in Rank1Processor._init().
# Hardcoded to 1 here regardless of what --tensor-parallel-size/-gpus is
# passed, until/unless that's re-investigated (e.g. via data parallelism -
# two independent single-GPU engines splitting the query set - instead of
# tensor parallelism). The other models below are single-process HF/pylate
# calls that don't shard across GPUs at all, so --tensor-parallel-size is
# simply ignored for them too (see _run_one).
# RaDeR bi-encoder retrievers (the size-ablation family from the same
# collection - 3B/7B/14B all share the recipe below; each card's own vLLM
# serving snippet confirms `pooling_type=LAST, normalize=true` for every
# size).
RADER_BIENCODER_MODELS = {
    "rader-3b": "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "rader-7b": "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "rader-14b": "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes",
}
# NOTE the repo-name inconsistency above is real and verified against the HF
# API: the 3B uses an underscore ("Qwen25_3B"), the 7B/14B a hyphen
# ("Qwen25-7B"/"Qwen25-14B").


def _build_rader_biencoder_processor(model_name: str):
    """LEGACY (reference-only since 2026-08-20): the production default for
    the RaDeR bi-encoders is now _build_rader_biencoder_vllm below - vLLM is
    each card's own recommended serving AND was verified to match raw-HF
    last-token embeddings exactly (scripts/test_vllm_feasibility.py), while
    this ST path needed two hard-won fixes (lasttoken pooling, chat-template
    modality stripping) to get there. Kept importable because the
    feasibility harness's frozen reference recipes mirror it.

    Raderspace/RaDeR_Qwen25-*: these repos ship NO sentence-transformers
    module config at all (no modules.json / config_sentence_transformers.json
    / 1_Pooling/config.json - confirmed by listing the actual repos, for all
    three sizes). Loading one via the generic `SentenceTransformer(name)`
    path therefore falls back to sentence-transformers' own default (mean
    pooling) - but each model's own card explicitly says to serve it via
    vLLM with `pooling_type=LAST, normalize=true`. Confirmed the hard way on
    the 14B: a full run under that wrong mean-pooling fallback scored
    statement-full nDCG@10=0.4126 against a reference value of 0.488, a far
    larger gap than every other model in this pipeline - and chunking (the
    earlier OOM fix) and precision were both ruled out as the cause (only
    0.32% of documents even exceed the 2048-token chunk cutoff). Building
    the Transformer+Pooling stack explicitly instead of relying on
    auto-detection, so lasttoken pooling actually gets used.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    transformer = modules.Transformer(
        model_name,
        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto"},
        config_kwargs={"trust_remote_code": True},
    )
    # Strip the auto-inferred "message" modality: because these Qwen2.5-based
    # repos ship a chat template in their tokenizer, sentence-transformers
    # (>=5.7, incl. the pinned 5.7.0) routes even plain strings through
    # apply_chat_template - a 13-token text becomes 39 tokens of
    # <|im_start|>system...<|im_end|> markup, so "lasttoken" pools the
    # template's trailing newline instead of the last content token.
    # Confirmed the hard way on 2026-08-19 via scripts/test_vllm_feasibility.py:
    # embeddings built this way rank at Spearman 0.19-0.39 against the true
    # last-token embeddings (raw HF forward == the models' own recommended
    # vLLM serving, cosine 1.0000 between those two), and stripping the
    # modality here restores exact agreement (cosine 1.0). Post-init
    # mutation deliberately, not a constructor modality_config - the
    # constructor path requires re-loading the full model a second time and
    # an explicit module_output_name.
    if "message" in getattr(transformer, "modality_config", {}):
        transformer.modality_config = {
            k: v for k, v in transformer.modality_config.items() if k != "message"
        }
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(), pooling_mode="lasttoken"
    )
    st = SentenceTransformer(modules=[transformer, pooling])
    return STProcessor(st, model_name)


def _build_rader_biencoder_vllm(model_name: str):
    """PRODUCTION path for the RaDeR bi-encoders since 2026-08-20: vLLM with
    the exact recipe validated against raw-HF last-token embeddings by
    scripts/test_vllm_feasibility.py (Spearman 0.9999-1.0 vs the fixed legacy
    reference). Every knob below is load-bearing:
    - pooling_type=LAST per each card's own vLLM serving snippet;
      normalize=False (not the card's normalize=true) because the production
      protocol mean-averages UN-normalized chunk vectors before the final
      cosine - see the legacy builder's history;
    - dtype=bfloat16 (since 2026-08-21): bf16 is RaDeR's TRAINING precision
      (their repo trains with --bf16; the checkpoint's torch_dtype float32
      is just storage format) and was validated FEASIBLE against the fixed
      legacy references (Spearman 0.9978-0.9994, |dNDCG@10| <= 0.019) while
      running up to ~16x faster end-to-end than fp32 on the 14B. The
      earlier "reduced precision wrecks rankings (Spearman 0.19-0.39)"
      finding that had forced fp32 was CONFOUNDED - it was measured against
      the chat-template-corrupted reference, and that mismatch was entirely
      the reference bug, not dtype. NOTE on provenance: the published
      2026-08-20 full nDCG runs were produced in fp32; the bf16 delta is
      below the bootstrap CI half-widths;
    - max_model_len=2080 (not 2048): chunk_to_context produces chunks of UP
      TO exactly 2048 tokens, and prompt_len == max_model_len sat right on
      the boundary where the engine wedge below was observed - the margin
      removes the boundary case outright;
    - enable_prefix_caching/enable_chunked_prefill=False: with vLLM's
      defaults (both True) the first full-benchmark rader runs WEDGED
      mid-statement-full (2026-08-20, jobs 727753-5 + the 730036 retry):
      client blocked forever in core_client.get_output, EngineCore alive but
      busy-looping with nothing scheduled, GPU at 0% until the idle reaper
      killed the job. Non-deterministic (retry passed the previously-hung
      query, then wedged two queries later), first appearing exactly where
      multi-chunk (2048-token) prompts start. The bert/e5 engines - which
      auto-run with both features off - completed every run cleanly, and
      neither feature helps embedding workloads (our EmbeddingProcessor
      caches per-text results anyway), so both are disabled rather than
      diagnosed further upstream."""
    return VLLMProcessor.from_huggingface(
        model_name,
        pooler_config={"pooling_type": "LAST", "normalize": False},
        dtype="bfloat16",
        max_model_len=2080,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )


def _build_inf_retriever_processor(model_name: str = "infly/inf-retriever-v1-pro"):
    """INF-Retriever-v1-Pro (7B, gte-Qwen2 lineage): auto-config
    SentenceTransformers load. The repo ships the full ST stack
    (modules.json: Transformer + lasttoken Pooling + Normalize) so unlike
    the RaDeR bi-encoders no hand-built module stack is needed - this IS
    the author-blessed path.

    Deliberate exception to the 2026-08-20 vLLM-default policy: the repo's
    bundled modeling_qwen.py runs BIDIRECTIONAL attention when used as an
    embedder (Qwen2Model.forward's is_causal defaults to False - checked in
    the repo source; same design as Alibaba's gte-Qwen2-7B-instruct this
    lineage descends from), while vLLM's stock Qwen2Model is causal. A
    candidate vLLM recipe (hf_overrides is_causal=False + LAST pooling) is
    registered in scripts/test_vllm_feasibility.py - switch this builder
    only after a FEASIBLE verdict there; running it causal-by-default would
    be exactly the silent wrong-attention/wrong-pooling class of bug that
    burned the RaDeR family.

    Load-bearing details:
    - trust_remote_code=True matters twice over: the bidirectional modeling
      AND the repo's tokenization_qwen.py, whose add_eos_token=true appends
      <|endoftext|> so "lasttoken" pooling lands on a constant summary
      position (a vanilla Qwen2Tokenizer silently ignores add_eos_token).
    - model_kwargs torch_dtype="auto" -> fp16, the checkpoint's declared
      dtype; ST's default load would silently compute in fp32, against the
      2026-08-21 precision-fairness policy (no model computes in fp32).
    - the tokenizer ships a chat template, so the sentence-transformers
      >=5.7 "message"-modality latch that chat-template-wrapped every RaDeR
      input (see _build_rader_biencoder_processor's history) applies here
      too; strip it post-init - a no-op if ST didn't infer it.
    - prompt-less on purpose: the repo defines a "query" instruction prompt
      (config_sentence_transformers.json) but default_prompt_name is null,
      and the benchmark protocol runs every model without instruction
      prefixes (instruction prompting is the team's separate experiment).
    - no chunking: max_seq_length is 32768 (sentence_bert_config.json) and
      essentially no benchmark document comes near it (only 0.32% exceed
      even 2048 tokens), so ST's silent truncation at 32768 is a no-op in
      practice.

    The actual ST construction lives in
    sabermath.processors.infx_retriever_processor.build_inf_retriever_st -
    promoted there (2026-08-21) so the INF-X-Retriever composition reuses
    the byte-identical retriever backend and the standalone-vs-INF-X
    comparison isolates the query rewrite alone.
    """
    from sabermath.processors import build_inf_retriever_st

    return STProcessor(build_inf_retriever_st(model_name), model_name)


CUSTOM_MODEL_BUILDERS = {
    "rank1-32b": lambda tp: Rank1Processor(tensor_parallel_size=1),
    # rank1 size ablations - same processor/prompt/scoring as rank1-32b,
    # just a different HF repo. tensor_parallel_size stays pinned to 1 (see
    # the rank1-32b note above; >1 would be pointless at these sizes anyway).
    "rank1-7b": lambda tp: Rank1Processor("jhu-clsp/rank1-7b"),
    "rank1-0.5b": lambda tp: Rank1Processor("jhu-clsp/rank1-0.5b"),
    # HF-transformers reference for the rank1 family. NOT production - it
    # exists so the vLLM-native path has something to be checked against
    # (rank1 was never migrated from HF, so it had no "before"). One
    # generation per pair at batch size 1; expect it to be orders of
    # magnitude slower than the vLLM path.
    # dtype ablation, NOT a separate model. rank1-32b's checkpoint is
    # bfloat16 and the production spec above pins vLLM to float16 (inherited
    # from the model card), so every production run logs "Casting
    # torch.bfloat16 to torch.float16". fp16's exponent range is far narrower
    # than bf16's, so this key runs the identical model at the checkpoint's
    # native precision to measure what that downcast costs. Same prompt, same
    # scoring, same tensor_parallel_size=1 - dtype is the ONLY difference.
    "rank1-32b-bf16": lambda tp: Rank1Processor(
        tensor_parallel_size=1, dtype="bfloat16"
    ),
    "rank1-32b-hf": lambda tp: Rank1HFProcessor("jhu-clsp/rank1-32b"),
    "rank1-7b-hf": lambda tp: Rank1HFProcessor("jhu-clsp/rank1-7b"),
    "rank1-0.5b-hf": lambda tp: Rank1HFProcessor("jhu-clsp/rank1-0.5b"),
    # Qwen3-Reranker family: vLLM-backed since 2026-08-20 (official model-card
    # recipe; verified FEASIBLE vs the HF path, Spearman >= 0.999 - see
    # scripts/test_vllm_feasibility.py). Qwen3RerankerProcessor remains the
    # legacy reference implementation.
    "qwen3-reranker-8b": lambda tp: Qwen3RerankerVLLMProcessor(),
    "qwen3-reranker-4b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-4B"
    ),
    "qwen3-reranker-0.6b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-0.6B"
    ),
    "gte-moderncolbert": lambda tp: ColBERTProcessor("lightonai/GTE-ModernColBERT-v1"),
    "reason-moderncolbert": lambda tp: ColBERTProcessor("lightonai/Reason-ModernColBERT"),
    # reasonir-8b: the official remote-code path (reverted from vLLM on
    # 2026-08-25). vLLM cannot reproduce the one thing that makes ReasonIR's
    # instruction mechanism work - its encode() runs a full bidirectional
    # pass over "instruction + text + embed_eos" and then zeroes the
    # instruction positions in the POOLING mask, which a stock MEAN pooler
    # has no way to express. Prompt-free the two agree to cosine 0.9999, so
    # this also restores the provenance of the published number, which
    # predates the vLLM switch. See REASONIR_INSTRUCTION_NOTE.
    "reasonir-8b": lambda tp: ReasonIRProcessor(),
    # p0-ONLY vLLM variant of reasonir-8b, added 2026-08-26. A SEPARATE key
    # rather than a change to the entry above, because the ST path must stay
    # for the instructed arms and for the published number's provenance.
    #
    # ReasonIR declares architectures: ["ReasonIRModel"], which vLLM does not
    # register - hence the hf_overrides redirect onto vLLM's own bidirectional
    # Llama, which is now known correct (it reproduced a genuinely
    # bidirectional reference for llama-embed-nemotron-8b to median |delta|
    # 6.9e-05 with zero verdict flips).
    #
    # MEASURED, not assumed (experiments/math-vs-word, 969 targets, via
    # scripts/repro_vllm_backend.py): against the ST reference built under
    # this model's own transformers==4.47.1 pin, this recipe agrees to
    # median |delta| 0.0012 with 2/969 verdict flips and an identical
    # math-vs-word statistic (66.46% both). That independently confirms the
    # "agree to cosine 0.9999 prompt-free" claim in the reasonir-8b comment
    # above - worth stating because scripts/test_vllm_feasibility.py, which
    # that claim cites, is not present in this repo.
    #
    # STRICTLY p0. See INSTRUCTION_EXCLUDED below: vLLM cannot express this
    # model's instruction mechanism (its encode() runs a bidirectional pass
    # over "instruction + text + embed_eos" and then zeroes the instruction
    # positions in the POOLING mask, which a stock MEAN pooler has no way to
    # represent), which is exactly why reasonir-8b was reverted off vLLM on
    # 2026-08-25. Instructed arms hard-error rather than silently returning
    # numbers from a mechanism that was not applied.
    "reasonir-8b-vllm": lambda tp: VLLMProcessor.from_huggingface(
        "reasonir/ReasonIR-8B",
        hf_overrides={
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        pooler_config={"pooling_type": "MEAN", "normalize": True},
    ),
    "splade-code-8b": lambda tp: SpladeProcessor(),
    # NOTE the 0.6B repo really is named "splade-code-06B" (no dot) -
    # verified against the HF API; "naver/splade-code-0.6B" does not exist.
    "splade-code-0.6b": lambda tp: SpladeProcessor("naver/splade-code-06B"),
    "rader-3b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-3b"]
    ),
    "rader-7b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-7b"]
    ),
    "rader-14b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-14b"]
    ),
    # RaDeR's pointwise cross-encoder reranker: vLLM-backed since 2026-08-20
    # (one-off LoRA merge; verified FEASIBLE, Spearman 0.9968 - see
    # RaDeRRerankerVLLMProcessor). RaDeRRerankerProcessor remains the legacy
    # reference implementation.
    "rader-reranker-7b": lambda tp: RaDeRRerankerVLLMProcessor(),
    # Diver's groupwise generative reranker (vLLM) - see GroupRankProcessor.
    # Unlike rank1-32b, tensor parallelism has no known score-corruption
    # history for this model, so --tensor-parallel-size is honored here; the
    # default single-GPU allocation still fits the 32B in bf16 on one H200.
    "diver-grouprank-32b": lambda tp: GroupRankProcessor(
        tensor_parallel_size=max(1, tp)
    ),
    # INF-Retriever-v1-Pro: SentenceTransformers-backed ON PURPOSE (the one
    # bi-encoder exception to the 2026-08-20 vLLM-default rollout) -
    # bidirectional remote-code attention with no VALIDATED vLLM path yet.
    # See _build_inf_retriever_processor.
    "inf-retriever-v1-pro": lambda tp: _build_inf_retriever_processor(),
    # INF-X-Retriever: the full infly SYSTEM the standalone model above was
    # built for - inf-query-aligner rewrites each query (official prompt +
    # the checkpoint's own sampling params, made reproducible via
    # per-query-content seeding), then the REWRITE is embedded with the
    # official instruct prefix against prefix-less documents, through the
    # byte-identical retriever backend as the standalone entry (so the two
    # rows isolate the rewrite+prefix effect). Recipe verified against the
    # team's own code (github.com/yaoyichen/INF-X-Retriever) - see
    # INFXRetrieverProcessor's module docstring for the full audit.
    # Deliberately NOT in scripts/test_vllm_feasibility.py: the harness
    # compares deterministic scoring paths of single models; this entry's
    # retriever half is already covered by inf-retriever-v1-pro's candidate
    # there, and its generative half has no reference to regress against.
    # Two 7B models co-resident on one GPU (~30GB in bf16+fp16) - fine on
    # an H200, no tensor parallelism. The rewrite log makes every generated
    # rewrite inspectable after the run (and warm-starts resumes; the
    # per-query seeding makes regeneration byte-identical either way) -
    # kept under the default save dir so the region sync ships it back to
    # the canonical host with the results.
    "inf-x-retriever": lambda tp: INFXRetrieverProcessor(
        rewrite_log_path="results/rerankers/.rewrites/inf-x-retriever.json"
    ),
}

# Custom models where --tensor-parallel-size is actually honored (everything
# else is either pinned to 1 for correctness - rank1-32b - or a single-GPU
# HF/PyLate call that can't shard at all).
CUSTOM_MODELS_USE_TP = {"diver-grouprank-32b"}

# Per-model scores_kwargs for CUSTOM_MODEL_BUILDERS entries (forwarded to
# processor.get_scores(**scores_kwargs) - see _run_one). The rader family
# keeps chunk_to_context/context_length=2048 (the paper's preprocessing
# protocol - only 0.32% of documents exceed 2048 tokens, but the numbers must
# stay preprocessing-identical across backends and sizes). batch_size was
# dropped when the family moved to vLLM (2026-08-20): it was an ST-side OOM
# guard; vLLM schedules/batches internally.
#
# The query_prompt/document_prompt/query_suffix/document_suffix entries are
# the vendor-documented input envelope for each model (see
# scripts/instructions/PROTOCOL.md): EmbeddingProcessor.get_scores applies
# them per side, before the vector cache, so an affixed text is simply a new
# cache key. --legacy strips every one of them, reproducing the pre-2026-08-25
# prompt-free runs.
#
# RaDeR: their retrievers.py builds "query: {instruction}{query}<|im_end|>"
# and "document: {doc}<|im_end|>" with last-token pooling; we sent raw text
# with neither marker nor EOS.
_RADER_BIENCODER_SCORES_KWARGS = {
    "chunk_to_context": True,
    "context_length": 2048,
    "query_prompt": "query: ",
    "document_prompt": "document: ",
    "query_suffix": RADER_EXPECTED_EOS,
    "document_suffix": RADER_EXPECTED_EOS,
}
# ReasonIR gets no text envelope: its own encode() adds nothing when the
# instruction is empty. The instructed arms go through per-side encode
# kwargs instead - see _experiment_scores_kwargs.
CUSTOM_MODEL_SCORES_KWARGS = {
    "rader-3b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-7b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-14b": dict(_RADER_BIENCODER_SCORES_KWARGS),
}

# Already supported through evaluate()'s generic HuggingFace path - no custom
# processor needed, just the right (model name, use_vllm) pair.
# tensor_parallel_size is forwarded via init_kwargs since
# VLLMProcessor.from_huggingface passes **kwargs straight through to
# vllm.LLM(...).
#
# All four generics are vLLM-backed since 2026-08-20 (previously only
# qwen3-embedding-8b was; the other three ran through the auto-detected
# SentenceTransformers path). The three Qwen3-Embedding derivatives get an
# explicit LAST+normalize pooler override as a plain dict (VLLMProcessor
# converts it version-tolerantly) - deterministic rather than trusting arch
# defaults, and verified FEASIBLE against the old ST path (Spearman 0.999-1.0,
# scripts/test_vllm_feasibility.py). trust_remote_code is
# VLLMProcessor.from_huggingface's default, so no init kwarg needed for it;
# the old ST-only model_kwargs/torch_dtype init kwargs would crash vllm.LLM
# and are gone with the backend.
GENERIC_MODELS = {
    "qwen3-embedding-8b": {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True},
    "reason-embed-qwen3-8b": {
        "model": "hanhainebula/reason-embed-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "diver-retriever-4b": {
        "model": "AQ-MedAI/Diver-Retriever-4B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    # Size-ablation sibling of diver-retriever-4b - identical init_kwargs so
    # the size comparison stays preprocessing-clean. dtype is pinned to
    # bfloat16 (2026-08-21): this repo's config declares no torch_dtype, so
    # vLLM's "auto" silently ran it in fp16 while the 4B sibling (bf16
    # config) ran bf16 - an inconsistent precision split within one
    # size-ablation family. Explicit bf16 validated against the frozen
    # legacy reference at Spearman 0.9989 (identical to the fp16 verdict).
    "diver-retriever-0.6b": {
        "model": "AQ-MedAI/Diver-Retriever-0.6B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "dtype": "bfloat16",
        },
    },
}
# rader-14b moved out of GENERIC_MODELS and into CUSTOM_MODEL_BUILDERS below -
# see _build_rader_14b_processor for why.
GENERIC_MODELS_USE_TP = {"qwen3-embedding-8b"}  # vLLM-backed -> tp actually used

ALL_MODEL_KEYS = list(CUSTOM_MODEL_BUILDERS) + list(GENERIC_MODELS)

_CHUNK512_SCORES_KWARGS = {"chunk_to_context": True, "context_length": 512}

TABLE_MODELS = {
    "qwen3-embedding-4b": {"model": "Qwen/Qwen3-Embedding-4B", "use_vllm": True},
    "qwen3-embedding-0.6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "use_vllm": True,
    },
    "harrier-oss-v1-270m": {
        "model": "microsoft/harrier-oss-v1-270m",
        "use_vllm": True,
    },
    "harrier-oss-v1-0.6b": {
        "model": "microsoft/harrier-oss-v1-0.6b",
        "use_vllm": True,
    },
    "harrier-oss-v1-27b": {
        "model": "microsoft/harrier-oss-v1-27b",
        "use_vllm": True,
    },
    "bge-m3": {"model": "BAAI/bge-m3", "use_vllm": True},
    # NOT given init_kwargs, and deliberately NOT moved off vLLM - both were
    # tried on 2026-08-26 and both were wrong. Full trail, because the
    # conclusion is counter-intuitive:
    #
    # Switching this model to vLLM changed its math-vs-word statistic by
    # -8.98pp (70.79% -> 61.82%, 107/969 verdict flips), which looked like a
    # vLLM defect. It is the opposite: vLLM is CORRECT and the previous
    # SentenceTransformers numbers were silently wrong.
    #
    # This repo declares architectures: ["LlamaBidirectionalModel"] with an
    # auto_map onto its own llama_bidirectional_model.py, which makes the
    # encoder bidirectional through ONE effective hook: an override of
    # LlamaModel._update_causal_mask() that returns no mask. (Its other hook,
    # layer.self_attn.is_causal = False, is inert - transformers never
    # forwards is_causal to the attention interface.) That override also
    # ASSERTS the attention impl is flash_attention_2 or eager.
    #
    # transformers 5.x DELETED _update_causal_mask: LlamaModel.forward calls
    # masking_utils.create_causal_mask directly. So under transformers 5.16.1
    # - which an unpinned install now resolves to - the override is never
    # called, the assert never fires, and the model runs CAUSAL while
    # reporting nothing. Every ST number for this model produced that way is
    # a causal run of a bidirectional model.
    #
    # Measured (experiments/math-vs-word, 969 targets, same subset):
    #   ST tf5.16.1 sdpa   (causal, as published)   78.95%
    #   vLLM fp32                                   68.02%
    #   ST tf4.51.3 eager  (genuinely bidirectional) 68.02%
    #   bidirectional ST vs vLLM : median |d| 8.2e-05, max 4.4e-04, 0 flips
    #   bidirectional ST vs published ST: median |d| 8.2e-02, 29 flips
    # i.e. vLLM reproduces the bidirectional reference to kernel precision.
    #
    # dtype was also ruled out: pinning float32 left the gap to the causal ST
    # run at median 0.0884 (bf16 gave 0.0889), so precision explained none of
    # it. No dtype pin is kept - vLLM's "auto" reads bfloat16 from the config,
    # and the bf16 and fp32 runs give the same 61.82%.
    #
    # DO NOT "fix" this by setting use_vllm: False without also pinning
    # transformers <5 AND forcing attn_implementation - ST alone silently
    # regresses to causal. Verify any change with
    # experiments/math-vs-word/scripts/repro_st_backend.py --attn eager,
    # which prints whether the hook is live before it scores.
    "llama-embed-nemotron-8b": {
        "model": "nvidia/llama-embed-nemotron-8b",
        "use_vllm": True,
    },
    "kalm-embedding-gemma3-12b-2511": {
        "model": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "use_vllm": True,
    },
    # EmbeddingGemma is a prompt-template-trained model: its documented
    # grammar is a fixed closed set, not free text.
    "embeddinggemma-300m": {
        "model": "google/embeddinggemma-300m",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "task: search result | query: ",
            "document_prompt": "title: none | text: ",
        },
    },
    # jina-v5 ships prompts {"query": "Query: ", "document": "Document: "}
    # with default_prompt_name="document" - so the prompt-less ST path used
    # to prepend "Document: " to QUERIES as well as documents.
    "jina-embeddings-v5-text-small": {
        "model": "jinaai/jina-embeddings-v5-text-small",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "Query: ",
            "document_prompt": "Document: ",
        },
    },
    "jina-embeddings-v5-text-nano": {
        "model": "jinaai/jina-embeddings-v5-text-nano",
        "use_vllm": False,
        "init_kwargs": {
            "trust_remote_code": True,
            "device": "cuda",
            "model_kwargs": {"dtype": "bfloat16", "default_task": "retrieval"},
        },
        "scores_kwargs": {
            "query_prompt": "Query: ",
            "document_prompt": "Document: ",
        },
        "disable_st_default_prompt": True,
    },
    # vLLM CONFIRMED CORRECT for this family (2026-08-26), and the pre-2026-08-26
    # SentenceTransformers numbers were the broken ones. Moving Octen-8B to vLLM
    # shifted its math-vs-word statistic +6.60pp (48.30% -> 54.90%, 136/969
    # verdict flips), which looked like a vLLM regression. It is not:
    #
    #   * dtype ruled out - fp32 vLLM matches bf16 vLLM to median 6.5e-04
    #     (2/969 flips), while the gap to ST stayed at 0.069;
    #   * pooling mode ruled out - vLLM matches ST[lasttoken] far better than
    #     ST[mean] (0.56) or ST[cls] (0.10), i.e. LAST is resolved correctly
    #     from the sentence_transformers config;
    #   * tokenization ruled out CONCLUSIVELY - the two backends emit
    #     byte-identical input ids, EOS 151643 included, and re-embedding
    #     through vLLM with explicit prompt_token_ids changes nothing
    #     (0.70380 via text vs 0.70351 via ids);
    #   * padding/batching ruled out - encoding one text at a time instead of
    #     in a padded batch changes nothing (0.99984 either way).
    #
    # The cause was the OLD loader. math-vs-word's _build_octen_processor
    # HAND-BUILT a [Transformer, Pooling(lasttoken), Normalize] stack, on the
    # premise that the generic SentenceTransformer load crashes on this repo's
    # 2_Normalize config. Measured on the same 14 probes:
    #
    #     hand-built ST stack vs vLLM : cosine 0.704
    #     GENERIC ST load     vs vLLM : cosine 0.99978
    #
    # The generic load works ON sentence-transformers 5.7.0 and agrees with
    # vLLM to 2e-04. It is version-dependent: on 6.0.0 it raises
    # "Normalize.__init__() got an unexpected keyword argument
    # 'normalize_embeddings'" and there is then NO correct ST option at all,
    # since the hand-built fallback is the broken one (0.68 there). So the hand-built imitation of the vendor stack was the
    # outlier, not vLLM - the same shape of finding as llama-embed-nemotron-8b
    # above, reached by a different route.
    #
    # Cross-check: Qwen/Qwen3-Embedding-4B, same architecture family, loaded
    # GENERICALLY, agrees with vLLM at 0.9998 - so vLLM's Qwen3 embedding path
    # is sound in general and this was never an Octen-specific vLLM problem.
    #
    # Reproduce with experiments/math-vs-word/scripts/diag_tokenization.py,
    # which loads the ST side the way each model's own path does and reports
    # which it used.
    "octen-embedding-4b": {"model": "Octen/Octen-Embedding-4B", "use_vllm": True},
    "octen-embedding-8b": {"model": "Octen/Octen-Embedding-8B", "use_vllm": True},
    "roberta-base": {
        "model": "FacebookAI/roberta-base",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": False},
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
    "bert-base-uncased": {
        "model": "google-bert/bert-base-uncased",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": False},
            "export_clean": True,
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
    # DELIBERATELY prompt-free, against the model card. E5 documents
    # "query: " / "passage: " as required ("otherwise you will see a
    # performance degradation"), but measured on this benchmark those
    # prefixes COST 0.028-0.080 nDCG, consistently, at p0, p1 and p2, on both
    # tasks that use problem+solution documents - and the effect is not a
    # chunking artifact (see FINDINGS_2026-08-25.md). We report our own
    # prompt-free configuration and disclose the deviation.
    "multilingual-e5-large": {
        "model": "intfloat/multilingual-e5-large",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": True},
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
}

API_MODELS = {
    "gemini-embedding-001": ("google", "gemini-embedding-001"),
    "gemini-embedding-2": ("google", "gemini-embedding-2"),
    "text-embedding-3-small": ("openrouter", "text-embedding-3-small"),
    "text-embedding-3-large": ("openrouter", "text-embedding-3-large"),
}

# Gemini's retrieval mechanism is the task_type enum, not a text prefix; it
# reaches embed_content(config=EmbedContentConfig(task_type=...)) through
# EmbeddingProcessor's per-side encode kwargs.
#
# gemini-embedding-001 ONLY. Live-probed 2026-08-25 against the real API:
# on -001, RETRIEVAL_QUERY returns byte-identical vectors to sending no
# config at all (it is the server-side default) while RETRIEVAL_DOCUMENT
# differs substantially (cosine 0.83-0.88 against the query-mode vector on
# benchmark-like text), so the parameter is live and only the DOCUMENT side
# actually changes. On gemini-embedding-2 every task_type - all eight valid
# values - returns a byte-identical vector, while an invalid value still
# 400s: the parameter is accepted and validated but has no effect on the
# embedding. Sending it there would buy nothing and would split one embed
# call into two (see EmbeddingProcessor._get_scores_per_side), so -2 is
# deliberately left with no envelope. Re-probe with
# scripts/instructions/diagnose_protocol.py gemini before assuming this is
# still true. text-embedding-3-* have no mechanism of any kind.
API_MODEL_SCORES_KWARGS = {
    "gemini-embedding-001": {
        "query_encode_kwargs": {"task_type": "RETRIEVAL_QUERY"},
        "document_encode_kwargs": {"task_type": "RETRIEVAL_DOCUMENT"},
    },
}

# The "-no-tok" keys are each method in its OWN default tokenization, with no
# mathematics-aware preprocessing: BM25 and Jaccard fall back to
# whitespace-delimited tokens, TF-IDF to scikit-learn's regex word tokenizer
# plus its own lowercasing. The plain keys use Approach Zero's pya0 tokenizer
# on every LaTeX block (see tokenization_helper.math_word_tokens).
#
# Both are reported: the paper's Table 1 lexical rows are the math-aware ones,
# and Table 3 carries the "-no-tok" rows beside them. The math tokenizer is
# NOT uniformly better - it gains BM25 (+0.018) and Jaccard (+0.009) but costs
# TF-IDF (-0.018) - which is exactly why both belong in the paper.
LEXICAL_MODEL_BUILDERS = {
    "bm25": BM25Processor,
    "tf-idf": TfidfProcessor,
    "jaccard": JaccardProcessor,
    "approach0": Approach0Processor,
    "bm25-no-tok": lambda: BM25Processor(tokenize_approach0=False),
    "tf-idf-no-tok": lambda: TfidfProcessor(tokenize_approach0=False),
    "jaccard-no-tok": lambda: JaccardProcessor(tokenize_approach0=False),
}

QWEN3_RERANKER_REPOS = {
    "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
    "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}

COLBERT_REPOS = {
    "gte-moderncolbert": "lightonai/GTE-ModernColBERT-v1",
    "reason-moderncolbert": "lightonai/Reason-ModernColBERT",
}


INSTRUCTION_EXCLUDED = {
    "reasonir-8b-vllm": (
        "vLLM cannot express ReasonIR's instruction mechanism - its encode() "
        "masks the instruction positions out of the POOLING mask, not out of "
        "the text, and a stock MEAN pooler has no way to represent that. Use "
        "the reasonir-8b key (ReasonIRProcessor) for p1/p2/p3; this key "
        "exists only to run p0 on vLLM"
    ),
    "tf-idf-no-tok": (
        "same reason as tf-idf: a document-fitted vocabulary plus cosine "
        "dilution makes instruction words pure noise"
    ),
    "jaccard-no-tok": (
        "same reason as jaccard: instruction tokens inflate the query "
        "token-set union"
    ),
    "tf-idf": (
        "its vocabulary is fitted on documents only and cosine scoring "
        "dilutes real query terms, so instruction words act as pure noise"
    ),
    "jaccard": (
        "instruction tokens inflate the query token-set union, distorting "
        "every score monotonically"
    ),
    "approach0": (
        "its _BROKEN_QUERIES md5 skip-list matches raw query text, so any "
        "query rewrite reintroduces known segfaults"
    ),
}

EXPERIMENT_MODEL_KEYS = (
    ALL_MODEL_KEYS
    + list(TABLE_MODELS)
    + list(API_MODELS)
    + list(LEXICAL_MODEL_BUILDERS)
)


@contextlib.contextmanager
def _result_lock(filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lock_path = filepath.with_name(filepath.name + ".lock")
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_result(
    filepath: Path,
    model_key: str,
    new_output: dict,
    extra_report_fields: dict | None = None,
) -> dict:
    with _result_lock(filepath):
        merged = _merge_results(filepath, model_key, new_output)
        if extra_report_fields:
            merged["reports"][model_key].update(extra_report_fields)
        tmp = filepath.with_name(filepath.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(filepath)
    return merged


def _write_meta(meta_path: Path, meta: dict) -> None:
    with _result_lock(meta_path):
        existing = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}
        merged = dict(existing)
        merged.update(meta)
        seen_tasks = list(existing.get("tasks", []))
        for task in meta.get("tasks", []):
            if task not in seen_tasks:
                seen_tasks.append(task)
        merged["tasks"] = seen_tasks
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(meta_path)


def _merge_results(filepath: Path, model_key: str, new_output: dict) -> dict:
    """Merge new_output into whatever's already saved at filepath (if
    anything), so writing one --task slice never erases another
    already-saved --task slice for the same model+subset (e.g. running
    statement-full and full-full for the same model as two separate
    submissions). Tasks are merged by name - re-running the same task
    overwrites just that task's entry with fresh data. A run that crashed
    entirely (no tasks computed) never overwrites previously-successful
    data for the same file."""
    if not filepath.exists():
        return new_output

    try:
        existing = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError):
        return new_output

    existing_report = existing.get("reports", {}).get(model_key)
    new_report = new_output["reports"][model_key]

    if existing_report is None:
        return new_output
    if "error" in new_report:
        # This attempt failed - keep whatever was already saved rather than
        # replacing good data with an error.
        return existing if "tasks" in existing_report else new_output
    if "error" in existing_report:
        return new_output

    existing_tasks = {t["task"]: t for t in existing_report.get("tasks", [])}
    for t in new_report.get("tasks", []):
        existing_tasks[t["task"]] = t
    existing_report["tasks"] = list(existing_tasks.values())

    existing_ndcgs = existing_report.get("ndcgs_by_task", {})
    existing_ndcgs.update(new_report.get("ndcgs_by_task", {}))
    existing_report["ndcgs_by_task"] = existing_ndcgs

    for key in ("model", "processor", "dcg_variant", "k"):
        if key in new_report:
            existing_report[key] = new_report[key]

    domains = new_output.get("domains") or existing.get("domains")
    merged = {"domains": domains, "reports": {model_key: existing_report}}
    for key in ("query_row_idxs", "part"):
        value = new_output.get(key)
        if value is None:
            value = existing.get(key)
        if value is not None:
            merged[key] = value
    return merged


def _partial_report_from_checkpoints(
    model_key: str, checkpoint_dir: Path, tasks: list[str], domains: list[list[str]]
) -> dict:
    """Rebuild an in-progress snapshot of this run's tasks straight from
    their checkpoint files (each {query_idx: ndcg_or_null}), without waiting
    for evaluate() to finish. Same {"task", "ndcg_at_k", "branches"} shape as
    a real TaskResult, plus "n_done"/"n_total" so a partial write is never
    mistaken for a finished one at a glance."""
    n_total = len(domains)
    tasks_out = []
    ndcgs_by_task = {}

    for task in tasks:
        ckpt_path = Path(checkpoint_dir) / f"{task}.json"
        if not ckpt_path.exists():
            continue
        try:
            raw = json.loads(ckpt_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        by_idx = {int(k): v for k, v in raw.items()}
        all_ndcgs = [by_idx.get(i) for i in range(n_total)]
        ndcgs_by_task[task] = all_ndcgs

        valid = [v for v in all_ndcgs if v is not None]
        ndcg_at_k = float(sum(valid) / len(valid)) if valid else 0.0

        branches_out = []
        for branch in BRANCHES:
            branch_vals = [
                by_idx[i]
                for i in range(n_total)
                if by_idx.get(i) is not None and branch in domains[i]
            ]
            branch_ndcg = float(sum(branch_vals) / len(branch_vals)) if branch_vals else 0.0
            branches_out.append({"branch": branch, "ndcg_at_k": branch_ndcg})

        tasks_out.append(
            {
                "task": task,
                "ndcg_at_k": ndcg_at_k,
                "branches": branches_out,
                "n_done": len(valid),
                "n_total": n_total,
            }
        )

    report_dict = {
        "model": model_key,
        "processor": None,
        "dcg_variant": "exponent",
        "k": 10,
        "tasks": tasks_out,
        "ndcgs_by_task": ndcgs_by_task,
    }
    return {"domains": domains, "reports": {model_key: report_dict}}


def _sync_back_if_needed(save_dir: str) -> None:
    """Push save_dir back to the canonical host over rsync, mirroring
    _common.sh's sync_back_to_canonical() - but callable from inside the
    query loop (via on_progress), not just once at process exit. A no-op
    unless NEEDS_REGION_SYNC=1 is set in the environment (i.e. this node's
    home storage isn't the canonical copy - see scripts/rerankers/*.slurm).
    Failures are logged, never raised - a flaky sync must not crash a run
    that's otherwise making progress."""
    if os.environ.get("NEEDS_REGION_SYNC") != "1":
        return
    host = os.environ.get("SABERMATH_CANONICAL_HOST", "hala")
    path = os.environ.get("SABERMATH_CANONICAL_PATH", "/home/ivo_petrov/sabermath")
    try:
        # rsync exit 23/24 mean "some source files vanished or could not be
        # read" - guaranteed here, because this very loop renames *.json.tmp
        # into place while the transfer runs. Everything else did transfer, so
        # those are successes; check=True would turn every one of them into a
        # spurious failure message and hide the real ones.
        completed = subprocess.run(
            [
                "rsync",
                "-auz",
                "--exclude=*.tmp",
                "--exclude=.*.??????",
                f"{save_dir}/",
                f"{host}:{path}/{save_dir}/",
            ],
            capture_output=True,
            timeout=300,
        )
        if completed.returncode not in (0, 23, 24):
            raise RuntimeError(
                f"rsync exit {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace')[-400:]}"
            )
    except Exception as e:
        print(f"[!!] Periodic sync-back to {host} failed (will retry next time): {e}")


def _subset_key(
    n: int | None, seed: int, query_shard: int | None, query_shards: int | None
) -> str:
    parts = []
    if n is not None:
        parts.append(f"n{n}_seed{seed}")
    if query_shards:
        parts.append(f"shard{query_shard}of{query_shards}")
    return "__".join(parts) if parts else "full"


def _select_shard(queries, domains, query_row_idxs, query_shard, query_shards):
    """Strided (i % shards == shard) query split, so every shard sees a mix
    of domains and difficulties rather than one contiguous block. Each shard
    writes its own result file and its own checkpoint dir - nothing is shared
    between shards, so they are safe to run concurrently on separate nodes or
    even separate clusters. scripts/merge_parts.py stitches them back into
    one result file."""
    idxs = [i for i in range(len(queries)) if i % query_shards == query_shard]
    if not idxs:
        raise ValueError(
            f"Shard {query_shard}/{query_shards} selected no queries."
        )
    global_idxs = (
        [query_row_idxs[i] for i in idxs]
        if query_row_idxs is not None
        else list(idxs)
    )
    return queries.select(idxs), [domains[i] for i in idxs], global_idxs


def _run_one(
    model_key: str,
    save_dir: str,
    tasks: list[str],
    n: int | None,
    seed: int,
    tensor_parallel_size: int,
    progress_every: int,
    legacy: bool = False,
) -> None:
    # Keyed by (model, query-subset) so a checkpoint - or a saved result -
    # from one --n/--seed combination is never mistaken for another subset:
    # evaluate_task() checkpoints by query *position*, not content, and a
    # 20-query smoke test must never land in the same file as a full run.
    subset_key = f"n{n}_seed{seed}" if n is not None else "full"
    suffix = "" if subset_key == "full" else f"__{subset_key}"
    if legacy:
        suffix = f"{suffix}__legacy"
    filepath = Path(save_dir) / f"{model_key}{suffix}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(save_dir)
        / ".checkpoints"
        / model_key
        / (f"{subset_key}__legacy" if legacy else subset_key)
    )

    if (
        tensor_parallel_size > 1
        and model_key not in GENERIC_MODELS_USE_TP
        and model_key not in CUSTOM_MODELS_USE_TP
    ):
        reason = (
            "is pinned to tensor_parallel_size=1 (confirmed to corrupt scores "
            "otherwise - see Rank1Processor._init())"
            if model_key == "rank1-32b"
            else "doesn't shard across GPUs in this framework"
        )
        print(
            f"[~] {model_key} {reason} - "
            f"--tensor-parallel-size {tensor_parallel_size} will be ignored "
            f"and only 1 GPU actually used."
        )

    success = True

    try:
        queries, documents = load_data()
        domains = list(queries["domains"])

        if n is not None:
            rng = random.Random(seed)
            idxs = sorted(rng.sample(range(len(queries)), min(n, len(queries))))
            queries = queries.select(idxs)
            domains = [domains[i] for i in idxs]

        # Every `progress_every` freshly-scored queries (across all tasks in
        # this run, combined - not per-task), snapshot current progress
        # straight from the checkpoint files into the real results file (via
        # _merge_results, so it never clobbers an already-finished sibling
        # task) and push it to the canonical host if this node isn't it. So
        # a hard kill with no chance to run an exit handler loses at most
        # `progress_every` queries' worth of results, not the whole run.
        progress_counter = {"n": 0}

        def on_progress() -> None:
            if progress_every <= 0:
                return
            progress_counter["n"] += 1
            if progress_counter["n"] % progress_every != 0:
                return
            partial = _partial_report_from_checkpoints(
                model_key, checkpoint_dir, tasks, domains
            )
            _write_result(filepath, model_key, partial)
            _sync_back_if_needed(save_dir)

        if model_key in CUSTOM_MODEL_BUILDERS:
            model = CUSTOM_MODEL_BUILDERS[model_key](tensor_parallel_size)
            report, ndcgs = evaluate(
                model,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                scores_kwargs=_model_scores_kwargs(model_key, legacy=legacy),
            )
        else:
            spec = dict(GENERIC_MODELS[model_key])
            model_name = spec.pop("model")
            # Start from this model's own fixed init kwargs (e.g.
            # trust_remote_code=True), then layer tensor_parallel_size on top
            # for the vLLM-backed ones - never the reverse, so a model-specific
            # kwarg can't accidentally get dropped by the tp branch.
            init_kwargs = dict(spec.pop("init_kwargs", {}))
            if model_key in GENERIC_MODELS_USE_TP and tensor_parallel_size > 1:
                init_kwargs["tensor_parallel_size"] = tensor_parallel_size
            # scores_kwargs flows to processor.get_scores(**scores_kwargs) -
            # e.g. rader-14b's chunk_to_context/context_length/batch_size, to
            # cap peak activation memory per encode() call (see the note on
            # its GENERIC_MODELS entry above).
            spec.pop("scores_kwargs", None)
            spec.pop("disable_st_default_prompt", None)
            scores_kwargs = _model_scores_kwargs(model_key, legacy=legacy)
            report, ndcgs = evaluate(
                model_name,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                init_kwargs=init_kwargs,
                scores_kwargs=scores_kwargs,
                **spec,
            )

        report_dict = report.to_dict()
        report_dict["ndcgs_by_task"] = ndcgs
        output = {"domains": domains, "reports": {model_key: report_dict}}

        print(f"\n=== {model_key} ===")
        for task in report_dict["tasks"]:
            print(f"  {task['task']:<20} nDCG@10 = {task['ndcg_at_k']:.4f}")
            for branch in task["branches"]:
                print(f"      {branch['branch']:<22} {branch['ndcg_at_k']:.4f}")

    except Exception as e:
        success = False
        tb = traceback.format_exc()
        output = {
            "domains": None,
            "reports": {
                model_key: {"model": model_key, "error": str(e), "traceback": tb}
            },
        }
        # Print the full traceback, not just str(e) - confirmed the hard way
        # that a bare message ("type 'array.array' is not subscriptable")
        # isn't enough to find the actual failure site without reproducing
        # it separately by hand. Goes to this job's .error log.
        print(f"\n[!!] {model_key} failed:")
        print(tb)

    _write_result(filepath, model_key, output)
    print(f"[+] Wrote {filepath}")
    _sync_back_if_needed(save_dir)

    if not success:
        # Propagate failure to main()'s exit-code check - previously this
        # process always exited 0 even when the model run itself failed
        # (only a hard crash before reaching this function's own try/except
        # would have been caught as a failure), so "N/N succeeded" was not
        # trustworthy. Comes after the write+sync above so the error record
        # (and traceback) still lands on disk either way.
        sys.exit(1)


def _experiment_spec(model_key: str) -> dict | None:
    if model_key in GENERIC_MODELS:
        return GENERIC_MODELS[model_key]
    if model_key in TABLE_MODELS:
        return TABLE_MODELS[model_key]
    return None


def _model_scores_kwargs(model_key: str, legacy: bool = False) -> dict:
    """Every per-model get_scores kwarg: preprocessing protocol
    (chunk_to_context/context_length) plus the vendor input envelope
    (query_prompt/document_prompt/suffixes/per-side API params). --legacy
    drops the envelope only, so the pre-2026-08-25 prompt-free runs stay
    reproducible from the same code."""
    if model_key in CUSTOM_MODEL_SCORES_KWARGS:
        kwargs = dict(CUSTOM_MODEL_SCORES_KWARGS[model_key])
    elif model_key in API_MODEL_SCORES_KWARGS:
        kwargs = dict(API_MODEL_SCORES_KWARGS[model_key])
    else:
        spec = _experiment_spec(model_key)
        kwargs = dict(spec.get("scores_kwargs", {})) if spec is not None else {}
    if legacy:
        kwargs = {k: v for k, v in kwargs.items() if k not in AFFIX_KEYS}
    return kwargs


def _experiment_scores_kwargs(
    model_key: str,
    instruction: str | None = None,
    legacy: bool = False,
) -> tuple[dict, bool]:
    """(scores_kwargs, wrap_instruction). wrap_instruction is False when the
    model's own envelope already carries the instruction, so the generic
    Instruct:/Query: wrap must not be applied on top of it (ReasonIR)."""
    kwargs = _model_scores_kwargs(model_key, legacy=legacy)
    wrap_instruction = True

    if not legacy and model_key == "reasonir-8b" and instruction is not None:
        # The official mechanism, exactly: prepend
        # "<|user|>\n{task}\n<|embed|>\n" to the query and then exclude
        # precisely those tokens from the mean pool. ReasonIRProcessor.encode
        # takes that whole prefix as its `instruction` argument and does the
        # masking itself, so it goes through the per-side encode kwargs
        # rather than as a text affix - and the generic Instruct:/Query: wrap
        # must not be applied on top of it.
        wrap_instruction = False
        kwargs["query_encode_kwargs"] = {
            "instruction": REASONIR_QUERY_INSTRUCTION_SLOT.format(
                instruction=instruction
            )
        }
        kwargs["document_encode_kwargs"] = {"instruction": ""}

    return kwargs, wrap_instruction


def _experiment_processor_slot(
    model_key: str, instruction_key: str, legacy: bool = False
) -> str:
    if model_key in QWEN3_RERANKER_REPOS:
        return f"qwen3__{instruction_key}"
    if legacy and instruction_key != "p0" and (
        model_key == "diver-grouprank-32b" or model_key in COLBERT_REPOS
    ):
        return "instructed"
    return "default"


def _build_spec_processor(
    spec: dict, model_key: str, tensor_parallel_size: int, legacy: bool = False
):
    spec = dict(spec)
    model_name = spec.pop("model")
    init_kwargs = dict(spec.pop("init_kwargs", {}))
    use_vllm = spec.pop("use_vllm", False)
    disable_st_default_prompt = spec.pop("disable_st_default_prompt", False)
    if not use_vllm:
        processor = STProcessor.from_huggingface(model_name, **init_kwargs)
        if disable_st_default_prompt and not legacy:
            # This model's prompts are applied explicitly per side via
            # scores_kwargs; leaving default_prompt_name set would make
            # sentence-transformers prepend its own default on top.
            st = getattr(processor, "_model", None)
            current = getattr(st, "default_prompt_name", None)
            if current is not None:
                print(
                    f"[~] {model_key}: clearing sentence-transformers "
                    f"default_prompt_name={current!r} (prompts applied per side)"
                )
                st.default_prompt_name = None
        return processor
    if model_key in GENERIC_MODELS_USE_TP and tensor_parallel_size > 1:
        init_kwargs["tensor_parallel_size"] = tensor_parallel_size
    return VLLMProcessor.from_huggingface(model_name, **init_kwargs)


def _build_experiment_processor(
    model_key: str,
    instruction_key: str,
    tensor_parallel_size: int,
    save_dir: str,
    legacy: bool = False,
):
    instruction = INSTRUCTIONS[instruction_key]
    if model_key in QWEN3_RERANKER_REPOS:
        # This family has a genuine <Instruct> slot, so p0 must fill it with
        # the VENDOR default to be a real "no task instruction" baseline; the
        # repo's production math instruction lives on as prompt key "pm".
        task_instruction = instruction
        if task_instruction is None:
            if legacy:
                return Qwen3RerankerVLLMProcessor(QWEN3_RERANKER_REPOS[model_key])
            task_instruction = VENDOR_QWEN3_RERANKER_INSTRUCTION
        return Qwen3RerankerVLLMProcessor(
            QWEN3_RERANKER_REPOS[model_key], task_instruction=task_instruction
        )
    if model_key == "diver-grouprank-32b":
        # The scaffold reserve has to be identical in every arm, or p0 and
        # p1-p3 differ in per-document token budget as well as in prompt.
        if legacy and instruction is None:
            return GroupRankProcessor(
                tensor_parallel_size=max(1, tensor_parallel_size)
            )
        return GroupRankProcessor(
            tensor_parallel_size=max(1, tensor_parallel_size),
            scaffold_reserve_tokens=EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE,
        )
    if model_key in COLBERT_REPOS:
        # Same reasoning as diver-grouprank: ColBERT pads queries to
        # query_length with mask tokens (query augmentation), so an arm-only
        # query_length changes the query representation even for identical
        # text. Uniform across arms; the checkpoint defaults (48 / 128) are
        # kept only under --legacy.
        if legacy and instruction is None:
            return ColBERTProcessor(COLBERT_REPOS[model_key])
        return ColBERTProcessor(
            COLBERT_REPOS[model_key],
            query_length=EXPERIMENT_COLBERT_QUERY_LENGTH,
        )
    if model_key == "inf-x-retriever":
        return INFXRetrieverProcessor(
            rewrite_log_path=str(
                Path(save_dir) / ".rewrites" / "inf-x-retriever.json"
            )
        )
    if model_key in CUSTOM_MODEL_BUILDERS:
        return CUSTOM_MODEL_BUILDERS[model_key](tensor_parallel_size)
    spec = _experiment_spec(model_key)
    if spec is not None:
        return _build_spec_processor(
            spec, model_key, tensor_parallel_size, legacy=legacy
        )
    if model_key in API_MODELS:
        kind, model_name = API_MODELS[model_key]
        if kind == "google":
            return GoogleProcessor(model_name)
        if kind == "openrouter":
            return OpenRouterEmbeddingProcessor(model_name)
        return OpenAIProcessor(model_name)
    if model_key in LEXICAL_MODEL_BUILDERS:
        return LEXICAL_MODEL_BUILDERS[model_key]()
    raise KeyError(f"Unknown experiment model key: {model_key}")


def _assert_envelope_supported(model_key: str, processor, scores_kwargs: dict) -> None:
    """The vendor input envelope is applied by EmbeddingProcessor.get_scores.
    Every processor that gets one below is a bi-encoder, but a future spec
    edit could point one at a cross-encoder whose get_scores(**kwargs) would
    silently swallow it - fail loudly instead."""
    envelope = [k for k in scores_kwargs if k in AFFIX_KEYS]
    if envelope and not isinstance(processor, EmbeddingProcessor):
        raise TypeError(
            f"{model_key}: input-envelope kwargs {envelope} are only applied "
            f"by EmbeddingProcessor.get_scores, but this model's processor is "
            f"{type(processor).__name__}."
        )


def _protocol_tag(legacy: bool, instruction_template: str) -> str:
    parts = []
    if legacy:
        parts.append("legacy")
    default_template = "legacy" if legacy else "canonical"
    if instruction_template != default_template:
        parts.append("nl2" if instruction_template == "legacy" else "nl1")
    return "-".join(parts)


def _run_one_experiment(
    model_key: str,
    save_dir: str,
    tasks: list[str],
    n: int | None,
    seed: int,
    tensor_parallel_size: int,
    progress_every: int,
    instruction_keys: list[str],
    save_scores: bool,
    legacy: bool = False,
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    part_name: str | None = None,
    query_shard: int | None = None,
    query_shards: int | None = None,
) -> None:
    subset_key = _subset_key(n, seed, query_shard, query_shards)
    suffix = "" if subset_key == "full" else f"__{subset_key}"
    tag = _protocol_tag(legacy, instruction_template)
    tag_suffix = f"__{tag}" if tag else ""
    part_suffix = f"__part-{part_name}" if part_name else ""

    queries, documents = load_data()
    domains = list(queries["domains"])
    query_row_idxs = None

    if n is not None:
        rng = random.Random(seed)
        query_row_idxs = sorted(
            rng.sample(range(len(queries)), min(n, len(queries)))
        )
        queries = queries.select(query_row_idxs)
        domains = [domains[i] for i in query_row_idxs]

    if query_shards:
        queries, domains, query_row_idxs = _select_shard(
            queries, domains, query_row_idxs, query_shard, query_shards
        )
        print(
            f"[~] Query shard {query_shard}/{query_shards}: "
            f"{len(queries)} queries."
        )

    instruction_placement = "column" if legacy else "query"

    processors: dict[str, object] = {}
    task_scores_by_prompt: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for instruction_key in instruction_keys:
        instruction_text = INSTRUCTIONS[instruction_key]
        scores_kwargs, wrap_instruction = _experiment_scores_kwargs(
            model_key, instruction_text, legacy=legacy
        )
        query_instruction = instruction_text
        if model_key in QWEN3_RERANKER_REPOS or not wrap_instruction:
            query_instruction = None

        prompt_block = {
            "key": instruction_key,
            "text": instruction_text,
            "query_instruction_applied": query_instruction is not None,
            "protocol": "legacy" if legacy else "canonical",
            "protocol_tag": tag,
            "instruction_template": instruction_template,
            "instruction_placement": instruction_placement,
            "mechanism": (
                "control"
                if model_key in INSTRUCTION_CONTROL_MODELS
                else "vendor-instruction"
            ),
            "control_reason": INSTRUCTION_CONTROL_REASONS.get(model_key),
            "input_envelope": {
                k: v for k, v in scores_kwargs.items() if k in AFFIX_KEYS
            },
        }

        filepath = (
            Path(save_dir)
            / f"{model_key}__{instruction_key}{tag_suffix}{suffix}{part_suffix}.json"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = (
            Path(save_dir)
            / ".checkpoints"
            / model_key
            / f"{instruction_key}{tag_suffix}__{subset_key}{part_suffix}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_meta(
            checkpoint_dir / "meta.json",
            {
                "model_key": model_key,
                "instruction_key": instruction_key,
                "instruction_text": instruction_text,
                "query_instruction_applied": query_instruction is not None,
                "n": n,
                "seed": seed,
                "query_row_idxs": query_row_idxs,
                "k": 10,
                "dcg_variant": "exponent",
                "tasks": tasks,
                "protocol_tag": tag,
                "part": part_name,
                "query_shard": query_shard,
                "query_shards": query_shards,
                "prompt": prompt_block,
            },
        )

        progress_counter = {"n": 0}

        def on_progress(
            _filepath=filepath, _checkpoint_dir=checkpoint_dir, _counter=progress_counter
        ) -> None:
            if progress_every <= 0:
                return
            _counter["n"] += 1
            if _counter["n"] % progress_every != 0:
                return
            partial = _partial_report_from_checkpoints(
                model_key, _checkpoint_dir, tasks, domains
            )
            partial["query_row_idxs"] = query_row_idxs
            partial["part"] = part_name
            _write_result(_filepath, model_key, partial)
            _sync_back_if_needed(save_dir)

        print(f"\n[~] {model_key} / {instruction_key} ...")

        try:
            slot = _experiment_processor_slot(
                model_key, instruction_key, legacy=legacy
            )
            if slot not in processors:
                processors[slot] = _build_experiment_processor(
                    model_key,
                    instruction_key,
                    tensor_parallel_size,
                    save_dir,
                    legacy=legacy,
                )
            model = processors[slot]

            _assert_envelope_supported(model_key, model, scores_kwargs)

            report, ndcgs = evaluate(
                model,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                instruction=query_instruction,
                instruction_template=instruction_template,
                instruction_placement=instruction_placement,
                save_scores=save_scores,
                scores_kwargs=scores_kwargs,
            )

            report_dict = report.to_dict()
            report_dict["ndcgs_by_task"] = ndcgs
            output = {
                "domains": domains,
                "query_row_idxs": query_row_idxs,
                "part": part_name,
                "reports": {model_key: report_dict},
            }

            task_scores_by_prompt[instruction_key] = {
                t["task"]: t["ndcg_at_k"] for t in report_dict["tasks"]
            }

            print(f"\n=== {model_key} / {instruction_key} ===")
            for task in report_dict["tasks"]:
                print(f"  {task['task']:<20} nDCG@10 = {task['ndcg_at_k']:.4f}")

        except Exception as e:
            failures.append(instruction_key)
            tb = traceback.format_exc()
            output = {
                "domains": None,
                "reports": {
                    model_key: {
                        "model": model_key,
                        "error": str(e),
                        "traceback": tb,
                    }
                },
            }
            print(f"\n[!!] {model_key} / {instruction_key} failed:")
            print(tb)

        _write_result(
            filepath, model_key, output, extra_report_fields={"prompt": prompt_block}
        )
        print(f"[+] Wrote {filepath}")
        _sync_back_if_needed(save_dir)

    if task_scores_by_prompt:
        p0_scores = task_scores_by_prompt.get("p0")
        print(f"\n=== {model_key}: per-prompt nDCG@10 summary ===")
        for instruction_key, by_task in task_scores_by_prompt.items():
            parts = []
            for task_name, score in by_task.items():
                delta = ""
                if (
                    instruction_key != "p0"
                    and p0_scores is not None
                    and task_name in p0_scores
                ):
                    delta = f" ({score - p0_scores[task_name]:+.4f})"
                parts.append(f"{task_name}={score:.4f}{delta}")
            print(f"  {instruction_key}: " + " | ".join(parts))

    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EXPERIMENT_MODEL_KEYS,
        default=None,
        help="Subset of models to run (default: all reranker-pipeline models "
        "- see the dependency isolation warning in this script's module "
        "docstring before doing that in a single shared environment). The "
        "embedding-table/API/lexical keys are valid only together with "
        "--instructions.",
    )
    parser.add_argument(
        "--instructions",
        nargs="+",
        choices=list(INSTRUCTIONS),
        default=None,
        help="Run the instruction-prompt experiment for these prompt keys "
        "(p0 = no instruction baseline). Switches output naming to "
        "<model>__<key>[__subset].json with per-(model,prompt) checkpoints; "
        "omit entirely for the legacy production behavior.",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Persist per-query raw candidate scores and the applied ranking "
        "to <checkpoint-dir>/<task>.scores.json (requires --instructions; "
        "use --instructions p0 for baseline score capture).",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        default=None,
        help="Run only this task (default: all 3, in one process per model). "
        "Combine with --part-name to run the three tasks as three concurrent "
        "jobs without them racing on one result file.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Reproduce the pre-2026-08-25 protocol exactly: no vendor input "
        "envelopes (query:/passage:, Query:/Document:, RaDeR EOS, ReasonIR "
        "wrappers, Gemini task_type), Qwen3-Reranker p0 back on the repo's "
        "own math instruction, ColBERT/GroupRank arm-dependent budgets, the "
        "double-newline Instruct: template, and the instruction mapped into "
        "the problem column before the task transform. Results are written "
        "with a __legacy tag so they can never overwrite current-protocol "
        "runs. See scripts/instructions/PROTOCOL.md.",
    )
    parser.add_argument(
        "--instruction-template",
        choices=list(INSTRUCTION_TEMPLATES),
        default=None,
        help="Instruct:/Query: wrapper (default: 'legacy' with --legacy, "
        "'canonical' - the vendors' single newline - otherwise). Set "
        "explicitly to isolate the template's effect from everything else.",
    )
    parser.add_argument(
        "--part-name",
        type=str,
        default=None,
        help="Write this run's results to <model>__<key>[...]__part-<NAME>.json "
        "with its own checkpoint directory, instead of the shared result "
        "file. Use it whenever several jobs cover different slices of the "
        "same (model, prompt) cell - one job per --task, or one per query "
        "shard - so nothing races or gets clobbered by the region rsync. "
        "Merge with scripts/merge_parts.py.",
    )
    parser.add_argument(
        "--query-shards",
        type=int,
        default=None,
        help="Split the query set into this many strided shards (i %% shards) "
        "so a slow model can be run as N concurrent jobs. Each shard keeps "
        "its own result file and checkpoints; merge with "
        "scripts/merge_parts.py.",
    )
    parser.add_argument(
        "--query-shard",
        type=int,
        default=None,
        help="Which shard index (0-based) this job evaluates.",
    )
    parser.add_argument("--save-to", type=str, default="results/rerankers")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only a random n-query subset (for smoke-testing before "
        "committing to a full ~1000-query run)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs to shard across for vLLM-backed models (rank1-32b, "
        "qwen3-embedding-8b). Ignored (with a warning) for the other 5 "
        "models, which are single-GPU only in this framework.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Every N freshly-scored queries (combined across all tasks in "
        "this run), snapshot current progress into the real results file "
        "and push it to the canonical host if this node isn't it (default: "
        "10). Lower it for extra safety on very slow models (e.g. rank1-32b) "
        "at the cost of more frequent rsync overhead; 0 disables snapshotting "
        "entirely (falls back to writing only when the run finishes/fails, "
        "the old behavior).",
    )
    args = parser.parse_args()

    models = args.models or ALL_MODEL_KEYS
    tasks = [args.task] if args.task else ALL_TASKS

    instruction_template = args.instruction_template or (
        "legacy" if args.legacy else DEFAULT_INSTRUCTION_TEMPLATE
    )

    if (args.query_shards is None) != (args.query_shard is None):
        parser.error("--query-shards and --query-shard must be used together.")
    if args.query_shards is not None:
        if args.query_shards < 2:
            parser.error("--query-shards must be >= 2.")
        if not 0 <= args.query_shard < args.query_shards:
            parser.error(
                f"--query-shard must be in [0, {args.query_shards - 1}]."
            )
        if args.instructions is None:
            parser.error("--query-shards requires --instructions.")
    if args.part_name is not None and args.instructions is None:
        parser.error("--part-name requires --instructions.")

    instruction_keys = None
    if args.instructions is not None:
        instruction_keys = list(dict.fromkeys(args.instructions))
        instructed = [k for k in instruction_keys if k != "p0"]
        if instructed:
            for model_key in models:
                if model_key in INSTRUCTION_EXCLUDED:
                    parser.error(
                        f"{model_key} cannot run instructed prompts "
                        f"({', '.join(instructed)}): "
                        f"{INSTRUCTION_EXCLUDED[model_key]}. "
                        "Only --instructions p0 is valid for it."
                    )
    else:
        if args.save_scores:
            parser.error(
                "--save-scores requires --instructions (use --instructions "
                "p0 for baseline score capture) so it can never resume a "
                "pre-capture production checkpoint."
            )
        for model_key in models:
            if (
                model_key not in CUSTOM_MODEL_BUILDERS
                and model_key not in GENERIC_MODELS
            ):
                parser.error(
                    f"{model_key} is only wired through the instruction "
                    "experiment - pass --instructions (p0 for a plain "
                    "baseline run)."
                )

    ctx = mp.get_context("spawn")
    failures = []

    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(
            f"# Running {model_key} ({i}/{len(models)}) | tasks={tasks}"
            + (f" | instructions={instruction_keys}" if instruction_keys else "")
            + f" | protocol={'legacy' if args.legacy else 'canonical'}"
            + f" | template={instruction_template}"
            + (f" | part={args.part_name}" if args.part_name else "")
            + (
                f" | shard={args.query_shard}/{args.query_shards}"
                if args.query_shards
                else ""
            )
        )
        print("#" * 60)

        if instruction_keys is None:
            p = ctx.Process(
                target=_run_one,
                args=(
                    model_key,
                    args.save_to,
                    tasks,
                    args.n,
                    args.seed,
                    args.tensor_parallel_size,
                    args.progress_every,
                    args.legacy,
                ),
            )
        else:
            p = ctx.Process(
                target=_run_one_experiment,
                args=(
                    model_key,
                    args.save_to,
                    tasks,
                    args.n,
                    args.seed,
                    args.tensor_parallel_size,
                    args.progress_every,
                    instruction_keys,
                    args.save_scores,
                    args.legacy,
                    instruction_template,
                    args.part_name,
                    args.query_shard,
                    args.query_shards,
                ),
            )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"[!!] Process for {model_key} exited with code {p.exitcode}")
            failures.append(model_key)

    print("\n" + "=" * 60)
    print(f"Done. {len(models) - len(failures)}/{len(models)} succeeded.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
