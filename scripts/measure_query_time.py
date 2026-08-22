"""Measure standardized average per-query task time across every model in
the SaberMath benchmark's results tables (rerankers, bi-encoder embedding
models - open and closed/API - and classical/lexical baselines).

METHODOLOGY (revised 2026-08-20 alongside the vLLM-default rollout; the
previous revision's "rerankers keep their native batching" and
"qwen3-embedding-8b deliberately timed on SentenceTransformers" rules are
superseded):

  1. Every model is timed on its PRODUCTION backend - the exact processor
     construction run_rerankers.py / run_model.py uses (the reranker-pipeline
     models are literally built through run_rerankers.py's own builder
     tables, so the two can never drift apart). Since 2026-08-20 that means
     vLLM for most neural models; see results/vllm_feasibility/summary.json
     for the per-model validation of those backends.

  2. All executions are standardized to SIMILAR PARALLELISM CONDITIONS:
     16 documents in flight / per processing step (BATCH_SIZE below), rather
     than each model's own tuned default. Concretely:
       - vLLM models: candidates are submitted in slices of 16 per
         embed/score/classify/generate call. CAVEAT: vLLM still
         micro-batches/schedules internally within a slice - this caps
         request-level parallelism at 16, it does not impose lockstep
         batches the way a padded HF forward does.
       - rank1-*: the production single batched generate over all ~150
         candidates is sliced into 16-prompt calls. CAVEAT: reasoning-chain
         lengths vary wildly, so each slice waits on its slowest member -
         sliced timing reads WORSE than production's one big call would.
       - diver-grouprank-32b: group_size=20 (docs per prompt) is the model's
         official SCORING protocol, not a batching knob - changing it
         changes scores, so it stays 20; only the generate calls over
         group-prompts are sliced at 16 (a single slice in practice).
       - HF/SentenceTransformers models (splade, colbert, jina-v5-nano):
         true batch_size=16 (splade's old 1/4 defaults were an OOM guard for
         the 8B on long docs - if 16 OOMs on a node, rerun at the largest
         working size and note it in the result).
       - Closed APIs: "16 documents in flight", client-side.
         text-embedding-3-* have no batch API (one HTTP request per doc) ->
         max_concurrency=16. gemini-embedding-001 -> one 16-doc request at a
         time (batch_size=16, max_concurrency=1, the closest analog to a
         local batch loop). gemini-embedding-2 -> the API forces 1 doc per
         request, so 16 concurrent single-doc requests (max_concurrency=16;
         GoogleProcessor honors an explicitly passed value exactly since
         2026-08-20). CAVEAT: the provider's server-side batching is opaque -
         only the client side is standardized.
       - Classical baselines: bm25/tf-idf/jaccard score in 16-doc slices
         against an index/vectorizer fit on the FULL candidate set
         (score_batch_size=16 - scores identical, verified; corpus statistics
         are never chunked). That is the PROTOCOL side; the actual
         parallelism bound for CPU methods is the 16-lane thread cap below.
         Per-doc worker threads were considered and rejected: jaccard is
         pure-Python set arithmetic (the GIL serializes threads), bm25/
         tf-idf per-doc scoring is microseconds of vectorized work (thread
         overhead would dominate the measurement), and either way it would
         time an implementation production never runs. approach0 is the one
         GENUINE EXCEPTION: it delegates to pya0's search engine, which
         exposes no per-doc scoring hook to slice - it runs unmodified.
       - CPU thread cap: scripts/timing/run_timing.slurm exports
         OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=16 for EVERY timing job, so no
         method (BLAS-backed classical scoring included) uses more than 16
         parallel CPU lanes - the CPU-side counterpart of the 16-doc GPU
         batch budget, applied uniformly.

  3. Embedding models (bi-encoder, open or closed/API) have their
     amortization effect removed: every one of the N_QUERIES sampled queries
     is scored from a cold cache (check_cache=False/update_cache=False) - no
     candidate document embedding is reused across queries.

  4. Everything runs on a single H200 GPU node (see
     scripts/timing/run_timing.slurm) - including the classical/closed-API
     models, which don't need the GPU themselves, for node/environment
     consistency.

  5. Model loading is excluded from the timed region. Only the time from
     "start scoring this query's candidates" to "have the final sorted
     ranking" is measured, per query. No warmup query - a model whose first
     call pays a one-time JIT/CUDA-graph cost shows that in its numbers.

Task is fixed to statement-full (query="statement" version, i.e. the bare
"problem" field; documents="full" version, i.e. "Problem: ...\n\nSolution:
...") - the paper's main setting, matching the nDCG results.

Usage:
    python scripts/measure_query_time.py --model rank1-32b
    python scripts/measure_query_time.py --model bm25 --n-queries 30
    python scripts/measure_query_time.py --model gemini-embedding-2
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sabermath.benchmark import transform
from sabermath.data import load_data
from sabermath.processors import (
    Approach0Processor,
    BM25Processor,
    ColBERTProcessor,
    EmbeddingProcessor,
    GoogleProcessor,
    JaccardProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    TfidfProcessor,
    VLLMProcessor,
)

# The reranker-pipeline models are built through run_rerankers.py's OWN
# production builder tables (single source of truth - point 1 of the
# methodology): rank1-*, diver-grouprank-32b, qwen3-reranker-*,
# rader-reranker-7b, reasonir-8b, rader-3b/7b/14b via CUSTOM_MODEL_BUILDERS,
# and qwen3-embedding-8b, reason-embed-qwen3-8b, diver-retriever-4b/0.6b via
# GENERIC_MODELS.
from run_rerankers import (  # noqa: E402
    CUSTOM_MODEL_BUILDERS as PRODUCTION_CUSTOM_BUILDERS,
    GENERIC_MODELS as PRODUCTION_GENERIC_MODELS,
)

# --- how many queries to average over per model. Also overridable via
# --n-queries. ---
N_QUERIES = 50

TASK_QUERY_VERSION = "statement"
TASK_DOC_VERSION = "full"
SEED = 42
BATCH_SIZE = 16  # the standardized documents-in-flight count (see docstring)


class OpenRouterEmbeddingProcessor(EmbeddingProcessor):
    """text-embedding-3-small/large via OpenRouter instead of OpenAI
    directly - requested explicitly (OpenRouter key supplied instead of an
    OpenAI one). sabermath.processors.OpenAIProcessor has no way to point
    at a custom base_url; this replicates its async batching/retry logic
    with a different base_url and the OpenRouter model-id convention
    (provider-prefixed, e.g. "openai/text-embedding-3-small" - confirmed
    directly: OpenRouter exposes an OpenAI-API-compatible /v1/embeddings
    endpoint, verified with a real embeddings.create() call before wiring
    this in)."""

    processor = "openrouter"

    def __init__(self, model_name: str, api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Please install openai to use it as a processor") from e

        client_args = {"base_url": "https://openrouter.ai/api/v1"}

        if api_key is not None:
            client_args["api_key"] = api_key
        elif os.getenv("OPENROUTER_API_KEY"):
            client_args["api_key"] = os.getenv("OPENROUTER_API_KEY")
        else:
            warnings.warn(
                "No OpenRouter API key was provided. Set OPENROUTER_API_KEY "
                "or pass api_key=... explicitly. This may cause "
                "authentication issues.",
                stacklevel=2,
            )

        self._model_name = model_name
        self._client = AsyncOpenAI(**client_args)

    @property
    def model(self) -> str:
        return self._model_name

    async def _encode_one(self, text, sem, *, retries: int = 4, **kwargs):
        last_error = None
        for attempt in range(retries):
            try:
                async with sem:
                    response = await self._client.embeddings.create(
                        model=f"openai/{self._model_name}", input=text, **kwargs
                    )
                return response.data[0].embedding
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"Failed to encode text after {retries} attempts."
        ) from last_error

    async def encode_async(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        max_concurrency: int = 20,
        retries: int = 3,
        **kwargs,
    ) -> np.ndarray:
        if max_concurrency <= 0:
            raise ValueError('"max_concurrency" must be >= 1')
        sem = asyncio.Semaphore(max_concurrency)
        coros = [self._encode_one(text, sem, retries=retries, **kwargs) for text in texts]
        results = await asyncio.gather(*coros)
        return np.asarray(results, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        retries: int = 9,
        max_concurrency: int = 20,
        **kwargs,
    ) -> np.ndarray:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.encode_async(
                    texts,
                    show_progress_bar=show_progress_bar,
                    retries=retries,
                    max_concurrency=max_concurrency,
                    **kwargs,
                )
            )
        raise RuntimeError(
            ".encode() can only be called from a synchronous context."
        )


def _production_custom(key: str):
    """The exact processor run_rerankers.py builds in production (tp=1)."""
    return lambda: PRODUCTION_CUSTOM_BUILDERS[key](1)


def _production_generic(key: str):
    """The exact processor benchmark.py's _make_processor builds for a
    GENERIC_MODELS entry (all four are use_vllm=True since 2026-08-20)."""
    spec = PRODUCTION_GENERIC_MODELS[key]
    assert spec.get("use_vllm"), f"{key} is no longer vLLM-backed - update timing"
    return lambda: VLLMProcessor.from_huggingface(
        spec["model"], **spec.get("init_kwargs", {})
    )


def _vllm(repo: str, **init_kwargs):
    """models.txt-style plain vLLM embedding load (run_model.py
    --driver vllm)."""
    return lambda: VLLMProcessor.from_huggingface(repo, **init_kwargs)


def _jina_nano_st():
    """The one remaining SentenceTransformers model (EuroBERT backbone,
    vLLM-infeasible) - exact production kwargs from scripts/models.txt,
    including default_task=retrieval (current jina remote code refuses to
    encode without a task at all)."""
    return STProcessor.from_huggingface(
        "jinaai/jina-embeddings-v5-text-nano",
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"dtype": "bfloat16", "default_task": "retrieval"},
    )


# Pooler/init recipes for the models.txt "--driver vllm" lines - keep in sync
# with scripts/models.txt (validated in scripts/test_vllm_feasibility.py).
_CHUNK512_POOLER = {"pooling_type": "MEAN", "normalize": False}

RERANKER_BUILDERS = {
    "rank1-32b": _production_custom("rank1-32b"),
    "rank1-7b": _production_custom("rank1-7b"),
    "rank1-0.5b": _production_custom("rank1-0.5b"),
    "qwen3-reranker-8b": _production_custom("qwen3-reranker-8b"),
    "qwen3-reranker-4b": _production_custom("qwen3-reranker-4b"),
    "qwen3-reranker-0.6b": _production_custom("qwen3-reranker-0.6b"),
    # splade/colbert stay on their (non-vLLM) production backends; batch
    # standardization happens at the constructor since their batching knobs
    # live there (recorded in the result JSON via builder_batching below).
    "splade-code-8b": lambda: SpladeProcessor(
        query_batch_size=BATCH_SIZE, document_batch_size=BATCH_SIZE
    ),
    "splade-code-0.6b": lambda: SpladeProcessor(
        "naver/splade-code-06B",
        query_batch_size=BATCH_SIZE,
        document_batch_size=BATCH_SIZE,
    ),
    "gte-moderncolbert": lambda: ColBERTProcessor(
        "lightonai/GTE-ModernColBERT-v1", encode_batch_size=BATCH_SIZE
    ),
    "reason-moderncolbert": lambda: ColBERTProcessor(
        "lightonai/Reason-ModernColBERT", encode_batch_size=BATCH_SIZE
    ),
    "rader-reranker-7b": _production_custom("rader-reranker-7b"),
    "diver-grouprank-32b": _production_custom("diver-grouprank-32b"),
}

EMBEDDING_BUILDERS = {
    # Reranker-pipeline bi-encoders: production builders, verbatim.
    "reasonir-8b": _production_custom("reasonir-8b"),
    "reason-embed-qwen3-8b": _production_generic("reason-embed-qwen3-8b"),
    "diver-retriever-4b": _production_generic("diver-retriever-4b"),
    "diver-retriever-0.6b": _production_generic("diver-retriever-0.6b"),
    "rader-3b": _production_custom("rader-3b"),
    "rader-7b": _production_custom("rader-7b"),
    "rader-14b": _production_custom("rader-14b"),
    # SentenceTransformers-backed in production ON PURPOSE (bidirectional
    # remote-code attention, no validated vLLM path yet - see
    # _build_inf_retriever_processor in run_rerankers.py). The "embedding"
    # category's batch_size=16 default applies cleanly: STProcessor.encode
    # forwards it to sentence-transformers' own batching. NOTE bf16-style
    # first-query kernel-compile inflation doesn't apply (fp16), but ST-path
    # first-query CUDA warmup still does - read medians, not just means.
    "inf-retriever-v1-pro": _production_custom("inf-retriever-v1-pro"),
    "qwen3-embedding-8b": _production_generic("qwen3-embedding-8b"),
    # models.txt "--driver vllm" table models (plain loads).
    "qwen3-embedding-4b": _vllm("Qwen/Qwen3-Embedding-4B"),
    "qwen3-embedding-0.6b": _vllm("Qwen/Qwen3-Embedding-0.6B"),
    "harrier-oss-v1-270m": _vllm("microsoft/harrier-oss-v1-270m"),
    "harrier-oss-v1-0.6b": _vllm("microsoft/harrier-oss-v1-0.6b"),
    "harrier-oss-v1-27b": _vllm("microsoft/harrier-oss-v1-27b"),
    "bge-m3": _vllm("BAAI/bge-m3"),
    "llama-embed-nemotron-8b": _vllm("nvidia/llama-embed-nemotron-8b"),
    "kalm-embedding-gemma3-12b-2511": _vllm("tencent/KaLM-Embedding-Gemma3-12B-2511"),
    "embeddinggemma-300m": _vllm("google/embeddinggemma-300m"),
    "jina-embeddings-v5-text-small": _vllm("jinaai/jina-embeddings-v5-text-small"),
    "octen-embedding-4b": _vllm("Octen/Octen-Embedding-4B"),
    "octen-embedding-8b": _vllm("Octen/Octen-Embedding-8B"),
    # models.txt chunked trio - pooler + 512 context, chunk kwargs come via
    # MODEL_SCORES_KWARGS below.
    "roberta-base": _vllm(
        "FacebookAI/roberta-base",
        pooler_config=dict(_CHUNK512_POOLER),
        max_model_len=512,
    ),
    "bert-base-uncased": _vllm(
        "google-bert/bert-base-uncased",
        pooler_config=dict(_CHUNK512_POOLER),
        export_clean=True,
        max_model_len=512,
    ),
    "multilingual-e5-large": _vllm(
        "intfloat/multilingual-e5-large",
        pooler_config={"pooling_type": "MEAN", "normalize": True},
        max_model_len=512,
    ),
    # The one ST holdout (EuroBERT backbone - vLLM-infeasible, see
    # scripts/models.txt).
    "jina-embeddings-v5-text-nano": _jina_nano_st,
}

CLOSED_API_BUILDERS = {
    "gemini-embedding-001": lambda: GoogleProcessor("gemini-embedding-001"),
    "gemini-embedding-2": lambda: GoogleProcessor("gemini-embedding-2"),
    # Via OpenRouter, not OpenAI directly - requested explicitly. See
    # OpenRouterEmbeddingProcessor's own docstring.
    "text-embedding-3-small": lambda: OpenRouterEmbeddingProcessor("text-embedding-3-small"),
    "text-embedding-3-large": lambda: OpenRouterEmbeddingProcessor("text-embedding-3-large"),
}

CLASSICAL_BUILDERS = {
    "bm25": lambda: BM25Processor(),
    "tf-idf": lambda: TfidfProcessor(),
    "jaccard": lambda: JaccardProcessor(),
    "approach0": lambda: Approach0Processor(),
}

ALL_BUILDERS = {
    **{k: (v, "reranker") for k, v in RERANKER_BUILDERS.items()},
    **{k: (v, "embedding") for k, v in EMBEDDING_BUILDERS.items()},
    **{k: (v, "closed_api") for k, v in CLOSED_API_BUILDERS.items()},
    **{k: (v, "classical") for k, v in CLASSICAL_BUILDERS.items()},
}

# --- per-model scoring kwargs implementing methodology points 2 and 3 ---

_NO_CACHE = {"check_cache": False, "update_cache": False}

# Category baselines...
_CATEGORY_SCORES_KWARGS = {
    # All current reranker backends expose a get_scores batch/slice knob
    # except splade/colbert (constructor-level, see RERANKER_BUILDERS) -
    # overridden to {} for those below.
    "reranker": {"batch_size": BATCH_SIZE},
    "embedding": {**_NO_CACHE, "batch_size": BATCH_SIZE},
    "closed_api": {**_NO_CACHE},  # per-model below - APIs differ
    "classical": {"score_batch_size": BATCH_SIZE},
}

# ...and per-model overrides (replace the category baseline entirely).
_MODEL_SCORES_KWARGS = {
    # Constructor-level batching only:
    "splade-code-8b": {},
    "splade-code-0.6b": {},
    "gte-moderncolbert": {},
    "reason-moderncolbert": {},
    # Production preprocessing protocol rides along with the batch slicing:
    "rader-3b": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 2048},
    "rader-7b": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 2048},
    "rader-14b": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 2048},
    "roberta-base": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 512},
    "bert-base-uncased": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 512},
    "multilingual-e5-large": {**_NO_CACHE, "batch_size": BATCH_SIZE, "chunk_to_context": True, "context_length": 512},
    # Closed APIs - "16 documents in flight", per-API (see docstring):
    "gemini-embedding-001": {**_NO_CACHE, "batch_size": BATCH_SIZE, "max_concurrency": 1},
    "gemini-embedding-2": {**_NO_CACHE, "batch_size": 1, "max_concurrency": BATCH_SIZE},
    "text-embedding-3-small": {**_NO_CACHE, "max_concurrency": BATCH_SIZE},
    "text-embedding-3-large": {**_NO_CACHE, "max_concurrency": BATCH_SIZE},
    # No per-doc scoring hook to slice (pya0 search engine) - the one genuine
    # exception to the 16-per-step protocol:
    "approach0": {},
}

# Batching applied at construction time rather than get_scores - recorded in
# the result JSON so no setting is invisible in the output.
_BUILDER_BATCHING_NOTE = {
    "splade-code-8b": f"query_batch_size={BATCH_SIZE}, document_batch_size={BATCH_SIZE} (constructor)",
    "splade-code-0.6b": f"query_batch_size={BATCH_SIZE}, document_batch_size={BATCH_SIZE} (constructor)",
    "gte-moderncolbert": f"encode_batch_size={BATCH_SIZE} (constructor)",
    "reason-moderncolbert": f"encode_batch_size={BATCH_SIZE} (constructor)",
    "diver-grouprank-32b": "group_size=20 kept (scoring protocol, not batching); generate calls sliced at 16",
    "approach0": "NOT sliceable: pya0 search engine, no per-doc scoring hook",
    "gemini-embedding-2": "API forces batch_size=1; 16 concurrent single-doc requests instead",
    "text-embedding-3-small": "no batch API; max_concurrency=16 as the parallelism analog",
    "text-embedding-3-large": "no batch API; max_concurrency=16 as the parallelism analog",
}


def _get_scores_kwargs(category: str, model_key: str) -> dict:
    if model_key in _MODEL_SCORES_KWARGS:
        return dict(_MODEL_SCORES_KWARGS[model_key])
    return dict(_CATEGORY_SCORES_KWARGS[category])


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _safe_query_idxs(queries) -> list[int]:
    """Every query index EXCEPT the 29 known query-side pya0 segfault
    triggers (see sabermath.processors.approach0_processor._BROKEN_QUERIES,
    confirmed via direct reproduction: these crash pya0.search()
    specifically when used as the QUERY, never as a document). Applied
    uniformly for EVERY model, not just approach0 - the whole point is
    that every model times the exact same set of queries; if approach0
    were the only one excluding these, the "same 50 queries for everyone"
    guarantee would silently break for it alone."""
    from sabermath.processors.approach0_processor import _BROKEN_QUERIES

    broken = set(_BROKEN_QUERIES)
    full_texts = transform(queries, "full", None)
    return [i for i in range(len(queries)) if _md5(full_texts[i]) not in broken]


def generate_query_sample(n_queries: int, seed: int, save_to: str) -> Path:
    """Sample N_QUERIES indices ONCE from the safe pool (every query index
    except the 29 known pya0 segfault triggers - excluded for every model,
    not just approach0, so the same sample is valid for all of them) and
    persist them to a file. Every model-timing run then reads this same
    file rather than sampling independently - critical, since two
    independent rng.sample() calls with the same seed but different
    population sizes/contents (e.g. one model's run happening to filter a
    few different indices than another's) do NOT produce the same actual
    indices. Run this once, before submitting any per-model timing jobs."""
    print("[~] Loading dataset...")
    queries, _documents = load_data()
    safe_idxs = _safe_query_idxs(queries)
    print(
        f"[~] {len(queries) - len(safe_idxs)} of {len(queries)} queries excluded "
        f"(known pya0/approach0 segfault triggers) - {len(safe_idxs)} eligible."
    )

    rng = random.Random(seed)
    query_idxs = sorted(rng.sample(safe_idxs, min(n_queries, len(safe_idxs))))

    save_dir = Path(save_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "query_sample.json"
    out_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_queries_requested": n_queries,
                "n_queries": len(query_idxs),
                "n_eligible": len(safe_idxs),
                "query_idxs": query_idxs,
            },
            indent=2,
        )
    )
    print(f"[+] Wrote {out_path} - {len(query_idxs)} shared query indices for every model.")
    return out_path


def _load_query_sample(save_to: str) -> dict:
    path = Path(save_to) / "query_sample.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist yet. Run this script once with "
            "--generate-sample first (before submitting any per-model "
            "timing jobs), so every model reads the exact same sampled "
            "query indices instead of each sampling independently."
        )
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Sample the shared N_QUERIES query indices once and write "
        "them to <save-to>/query_sample.json, then exit - no --model "
        "needed. Run this exactly once before submitting any per-model "
        "timing jobs; every one of them reads this same file so every "
        "model is timed on the identical set of queries.",
    )
    parser.add_argument("--model", choices=sorted(ALL_BUILDERS))
    parser.add_argument(
        "--n-queries",
        type=int,
        default=N_QUERIES,
        help=f"Only used with --generate-sample (default: {N_QUERIES}, "
        "itself a placeholder - see N_QUERIES at the top of this file).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Only used with --generate-sample.")
    parser.add_argument("--save-to", type=str, default="results/timing")
    args = parser.parse_args()

    if args.generate_sample:
        generate_query_sample(args.n_queries, args.seed, args.save_to)
        return

    if not args.model:
        raise SystemExit("--model is required unless --generate-sample is passed")

    if args.n_queries != N_QUERIES or args.seed != SEED:
        print(
            f"[!!] --n-queries/--seed are ignored for a --model run (got "
            f"n_queries={args.n_queries}, seed={args.seed}) - every model "
            f"reads the shared query_sample.json instead. Re-run "
            f"--generate-sample first if you actually want different "
            f"values, then re-run every --model job so they all still "
            f"agree on the same sample."
        )

    builder, category = ALL_BUILDERS[args.model]

    print("[~] Loading dataset...")
    queries, documents = load_data()

    sample = _load_query_sample(args.save_to)
    query_idxs = sample["query_idxs"]
    print(
        f"[~] Loaded {len(query_idxs)} shared query indices from "
        f"query_sample.json (seed={sample['seed']})"
    )

    print(f"[~] Building processor for {args.model} ({category})...")
    processor = builder()

    scores_kwargs = _get_scores_kwargs(category, args.model)
    print(f"[~] get_scores kwargs for this run: {scores_kwargs}")
    if args.model in _BUILDER_BATCHING_NOTE:
        print(f"[~] builder-level batching: {_BUILDER_BATCHING_NOTE[args.model]}")

    per_query_seconds = []
    measured_idxs = []
    print(
        f"[~] Timing {len(query_idxs)} queries "
        f"(task={TASK_QUERY_VERSION}-{TASK_DOC_VERSION}, no warmup)..."
    )
    for qi in query_idxs:
        row = queries[qi]
        query_text = transform(queries, TASK_QUERY_VERSION, [qi])[0]
        doc_ids = list(row["candidates"])
        document_texts = transform(documents, TASK_DOC_VERSION, doc_ids)

        _sync_cuda()
        t0 = time.perf_counter()
        scores = processor.get_scores(
            query_text, document_texts, show_progress_bar=False, **scores_kwargs
        )
        if scores is None:
            # Approach0Processor's own defensive None return for a broken
            # query - shouldn't be reachable given the pre-filtering above,
            # but don't silently record a fake near-zero time if it is.
            _sync_cuda()
            print(f"[!!] query {qi}: get_scores() returned None, skipping")
            continue
        _ = np.argsort(-np.asarray(scores, dtype=float))  # the final ranking
        _sync_cuda()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        per_query_seconds.append(elapsed)
        measured_idxs.append(qi)
        print(f"    query {qi:4d} | {elapsed:.4f}s", flush=True)

    arr = np.asarray(per_query_seconds, dtype=float)
    result = {
        "model": args.model,
        "category": category,
        "backend": getattr(processor, "processor", None),
        "task": f"{TASK_QUERY_VERSION}-{TASK_DOC_VERSION}",
        "n_queries_requested": sample["n_queries_requested"],
        "n_queries_measured": len(per_query_seconds),
        "seed": sample["seed"],
        "query_idxs": measured_idxs,
        "standardized_batch_size": BATCH_SIZE,
        "scores_kwargs": scores_kwargs,
        "builder_batching": _BUILDER_BATCHING_NOTE.get(args.model),
        "per_query_seconds": per_query_seconds,
        "mean_seconds": float(arr.mean()) if len(arr) else None,
        "median_seconds": float(np.median(arr)) if len(arr) else None,
        "std_seconds": float(arr.std(ddof=1)) if len(arr) > 1 else None,
    }

    save_dir = Path(args.save_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.model}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[+] Wrote {out_path}")
    if result["mean_seconds"] is not None:
        print(
            f"[+] mean={result['mean_seconds']:.4f}s "
            f"median={result['median_seconds']:.4f}s "
            f"over {result['n_queries_measured']} queries"
        )
    else:
        print("[!!] No queries were successfully measured.")


if __name__ == "__main__":
    main()
