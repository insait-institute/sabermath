"""Evaluate the SABER-Math rerankers that need custom scoring logic (rank1
0.5B/7B/32B, Qwen3-Reranker 0.6B/4B/8B, GTE-ModernColBERT,
Reason-ModernColBERT, ReasonIR, SPLADE-code 0.6B/8B, RaDeR bi-encoders
3B/7B/14B, RaDeR-reranker-7B, Diver-GroupRank-32B, INF-Retriever-v1-Pro),
plus the models that run through the generic HuggingFace path
(Qwen3-Embedding-8B, Reason-Embed-Qwen3-8B, Diver-Retriever 0.6B/4B),
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
(SparseEncoder, env_splade.yml), and inf-retriever-v1-pro
(SentenceTransformers production path, env_inf_retriever.yml - its
bidirectional remote-code attention has no VALIDATED vLLM recipe yet, see
_build_inf_retriever_processor, and that remote code needs
transformers==4.51.0 exactly, see the env's header).
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
from sabermath.processors import (
    ColBERTProcessor,
    GroupRankProcessor,
    Qwen3RerankerProcessor,
    Qwen3RerankerVLLMProcessor,
    RaDeRRerankerProcessor,
    RaDeRRerankerVLLMProcessor,
    Rank1Processor,
    ReasonIRProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    VLLMProcessor,
)
from sabermath.schemas import Branch, Task

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

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
    """
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "auto"},
    )
    transformer = st[0]
    if "message" in getattr(transformer, "modality_config", {}):
        transformer.modality_config = {
            k: v for k, v in transformer.modality_config.items() if k != "message"
        }
    return STProcessor(st, model_name)


CUSTOM_MODEL_BUILDERS = {
    "rank1-32b": lambda tp: Rank1Processor(tensor_parallel_size=1),
    # rank1 size ablations - same processor/prompt/scoring as rank1-32b,
    # just a different HF repo. tensor_parallel_size stays pinned to 1 (see
    # the rank1-32b note above; >1 would be pointless at these sizes anyway).
    "rank1-7b": lambda tp: Rank1Processor("jhu-clsp/rank1-7b"),
    "rank1-0.5b": lambda tp: Rank1Processor("jhu-clsp/rank1-0.5b"),
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
    # reasonir-8b: vLLM-backed since 2026-08-20. vllm==0.26.0 has no
    # ReasonIRModel implementation, but ReasonIR IS a bidirectional-attention
    # Llama with mean pooling - exactly vLLM's LlamaBidirectionalModel
    # (llama-embed-nemotron's arch), so the architecture name is remapped and
    # the "pooling" config attr that impl reads is injected. NEAR-MATCH vs
    # the remote-code HF path (Spearman 0.9913, |dNDCG@10| 0.0014) - the
    # published full-run number predates this switch (legacy provenance).
    "reasonir-8b": lambda tp: VLLMProcessor.from_huggingface(
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
_RADER_BIENCODER_SCORES_KWARGS = {
    "chunk_to_context": True,
    "context_length": 2048,
}
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
    return {"domains": domains, "reports": {model_key: existing_report}}


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
        subprocess.run(
            ["rsync", "-auz", f"{save_dir}/", f"{host}:{path}/{save_dir}/"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as e:
        print(f"[!!] Periodic sync-back to {host} failed (will retry next time): {e}")


def _run_one(
    model_key: str,
    save_dir: str,
    tasks: list[str],
    n: int | None,
    seed: int,
    tensor_parallel_size: int,
    progress_every: int,
) -> None:
    # Keyed by (model, query-subset) so a checkpoint - or a saved result -
    # from one --n/--seed combination is never mistaken for another subset:
    # evaluate_task() checkpoints by query *position*, not content, and a
    # 20-query smoke test must never land in the same file as a full run.
    subset_key = f"n{n}_seed{seed}" if n is not None else "full"
    suffix = "" if subset_key == "full" else f"__{subset_key}"
    filepath = Path(save_dir) / f"{model_key}{suffix}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(save_dir) / ".checkpoints" / model_key / subset_key

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
            merged = _merge_results(filepath, model_key, partial)
            filepath.write_text(json.dumps(merged, indent=2))
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
                scores_kwargs=CUSTOM_MODEL_SCORES_KWARGS.get(model_key, {}),
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
            scores_kwargs = dict(spec.pop("scores_kwargs", {}))
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

    output = _merge_results(filepath, model_key, output)
    filepath.write_text(json.dumps(output, indent=2))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        default=None,
        help="Subset of models to run (default: all of them - see the "
        "dependency isolation warning in this script's module docstring "
        "before doing that in a single shared environment).",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        default=None,
        help="Run only this task (default: all 3, in one process per model)",
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

    ctx = mp.get_context("spawn")
    failures = []

    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(f"# Running {model_key} ({i}/{len(models)}) | tasks={tasks}")
        print("#" * 60)

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
