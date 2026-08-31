#!/usr/bin/env python3
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

# Must be set before numpy/torch, which read these once at import and cache
# the pool size. Every published timing number was measured with 16 CPU lanes,
# so a run on a machine with more cores would not be comparable.
TIMING_THREADS = os.environ.get("SABERMATH_TIMING_THREADS", "16")
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, TIMING_THREADS)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    RetroStarRewrittenProcessor,
    SpladeProcessor,
    TfidfProcessor,
    VLLMProcessor,
)

# Built through the registry's own production builders, so a timing number
# can never be measured on a backend the benchmark does not score with.
from sabermath.registry import (  # noqa: E402
    CUSTOM_MODEL_BUILDERS as PRODUCTION_CUSTOM_BUILDERS,
    GENERIC_MODELS as PRODUCTION_GENERIC_MODELS,
)

USAGE = """\
  python scripts/run_timing.py --model rank1-32b
  python scripts/run_timing.py --model bm25 --n-queries 30
  python scripts/run_timing.py --model gemini-embedding-2

Every model is timed on its production backend, at 16 documents in flight and
16 CPU lanes, on the statement-full task. The full measurement protocol - and
what makes one number comparable to another - is in docs/experiment-timing.md.
"""

# Queries averaged over per model; overridable with --n-queries.
N_QUERIES = 50

TASK_QUERY_VERSION = "statement"
TASK_DOC_VERSION = "full"
SEED = 42
BATCH_SIZE = 16  # the standardized documents-in-flight count (see docstring)


class OpenRouterEmbeddingProcessor(EmbeddingProcessor):

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
    return lambda: PRODUCTION_CUSTOM_BUILDERS[key](1)


def _production_generic(key: str):
    spec = PRODUCTION_GENERIC_MODELS[key]
    assert spec.get("use_vllm"), f"{key} is no longer vLLM-backed - update timing"
    return lambda: VLLMProcessor.from_huggingface(
        spec["model"], **spec.get("init_kwargs", {})
    )


def _vllm(repo: str, **init_kwargs):
    return lambda: VLLMProcessor.from_huggingface(repo, **init_kwargs)


def _jina_nano_st():
    return STProcessor.from_huggingface(
        "jinaai/jina-embeddings-v5-text-nano",
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"dtype": "bfloat16", "default_task": "retrieval"},
    )


# Keep in sync with src/sabermath/registry.py.
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
    "retro-star-32b": _production_custom("retro-star-32b"),
    # Timed with the rewrite CACHED, which the model forces: its production
    # build hands the 32B reranker 0.85 of the GPU precisely because the 7B
    # generator never loads. Read this as "reranking cost GIVEN a rewrite";
    # the generation term is measured separately as
    # reason-rewriter-reason-embed-8b.
    "retro-star-32b-rewritten": _production_custom("retro-star-32b-rewritten"),
    # End-to-end variant: the rewrite is generated inside the timed region,
    # which needs the co-resident memory split rather than the production one.
    # Read it as "both halves on one GPU", NOT as a drop-in for the cached row
    # above - the reranker half here has far less KV cache, so it schedules
    # fewer candidates concurrently and its reranking component is inflated.
    "retro-star-32b-rewritten-uncached": lambda: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-32b-0928",
        tensor_parallel_size=1,
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
        gpu_memory_utilization=0.55,
        rewriter_gpu_memory_utilization=0.30,
        require_cached_rewrites=False,
    ),
}

EMBEDDING_BUILDERS = {
    # Reranker-pipeline bi-encoders: production builders, verbatim.
    "reasonir-8b": _production_custom("reasonir-8b"),
    "reason-embed-qwen3-8b": _production_generic("reason-embed-qwen3-8b"),
    "reason-embed-llama-3.1-8b": _production_generic("reason-embed-llama-3.1-8b"),
    # Composed rewrite systems. The embedding category's _NO_CACHE is
    # load-bearing: check_cache governs the REWRITE cache as well as the vector
    # cache, so turning it off puts generation inside the timed region. A
    # cached run would time a dict lookup and report a composed system as
    # costing the same as its embedder half.
    "inf-x-retriever": _production_custom("inf-x-retriever"),
    "reason-rewriter-reason-embed-8b": _production_custom("reason-rewriter-reason-embed-8b"),
    "reason-rewriter-reason-embed-llama-3.1-8b": _production_custom(
        "reason-rewriter-reason-embed-llama-3.1-8b"
    ),
    "diver-retriever-4b": _production_generic("diver-retriever-4b"),
    "diver-retriever-0.6b": _production_generic("diver-retriever-0.6b"),
    "rader-3b": _production_custom("rader-3b"),
    "rader-7b": _production_custom("rader-7b"),
    "rader-14b": _production_custom("rader-14b"),
    # SentenceTransformers in production on purpose - see the registry. Its
    # first query still pays CUDA warmup, so read medians, not means.
    "inf-retriever-v1-pro": _production_custom("inf-retriever-v1-pro"),
    "qwen3-embedding-8b": _production_generic("qwen3-embedding-8b"),
    # Registry table models on plain vLLM loads.
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
    # The chunked trio - pooler + 512 context, chunk kwargs come via
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
    # src/sabermath/registry.py).
    "jina-embeddings-v5-text-nano": _jina_nano_st,
}

CLOSED_API_BUILDERS = {
    "gemini-embedding-001": lambda: GoogleProcessor("gemini-embedding-001"),
    "gemini-embedding-2": lambda: GoogleProcessor("gemini-embedding-2"),
    # Via OpenRouter, not OpenAI directly.
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


_NO_CACHE = {"check_cache": False, "update_cache": False}

# Category baselines...
_CATEGORY_SCORES_KWARGS = {
    # Every reranker backend exposes a get_scores batch knob except
    # splade/colbert, which take it at construction instead.
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
    # pya0 exposes no per-document scoring hook, so it is the one genuine
    # exception to the 16-per-step protocol.
    "retro-star-32b-rewritten-uncached": {**_NO_CACHE, "batch_size": BATCH_SIZE},
    "approach0": {},
}

# Batching applied at construction rather than get_scores; recorded in the
# result JSON so no setting is invisible in the output.
_BUILDER_BATCHING_NOTE = {
    "splade-code-8b": f"query_batch_size={BATCH_SIZE}, document_batch_size={BATCH_SIZE} (constructor)",
    "splade-code-0.6b": f"query_batch_size={BATCH_SIZE}, document_batch_size={BATCH_SIZE} (constructor)",
    "gte-moderncolbert": f"encode_batch_size={BATCH_SIZE} (constructor)",
    "reason-moderncolbert": f"encode_batch_size={BATCH_SIZE} (constructor)",
    "diver-grouprank-32b": "group_size=20 kept (scoring protocol, not batching); generate calls sliced at 16",
    "retro-star-32b-rewritten-uncached": "composed END-TO-END: 5 rewrites generated per query INSIDE the timed region; co-resident split 0.55 reranker + 0.30 rewriter (reranking half slower than production 0.85); generate calls sliced at 16",
    "retro-star-32b-rewritten": "composed: rewrite READ FROM CACHE (32B holds 0.85 of the GPU; generator cannot co-load) - excludes generation, add reason-rewriter-reason-embed-8b for end-to-end; generate calls sliced at 16",
    "inf-x-retriever": "composed: query rewrite generated per query (uncached, inside the timed region); documents sliced at 16",
    "reason-rewriter-reason-embed-8b": "composed: 5 rewrites generated per query (uncached, inside the timed region), embeddings mean-pooled; documents sliced at 16",
    "reason-rewriter-reason-embed-llama-3.1-8b": "composed: 5 rewrites generated per query (uncached, inside the timed region), embeddings mean-pooled; documents sliced at 16",
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
    from sabermath.processors.approach0_processor import _BROKEN_QUERIES

    broken = set(_BROKEN_QUERIES)
    full_texts = transform(queries, "full", None)
    return [i for i in range(len(queries)) if _md5(full_texts[i]) not in broken]


def generate_query_sample(n_queries: int, seed: int, save_to: str) -> Path:
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


def time_one_model(model_key: str, args, queries, documents, sample, query_idxs) -> None:
    builder, category = ALL_BUILDERS[model_key]

    print(f"[~] Building processor for {model_key} ({category})...")
    processor = builder()

    scores_kwargs = _get_scores_kwargs(category, model_key)
    print(f"[~] get_scores kwargs for this run: {scores_kwargs}")
    if model_key in _BUILDER_BATCHING_NOTE:
        print(f"[~] builder-level batching: {_BUILDER_BATCHING_NOTE[model_key]}")

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
            # Approach0Processor's None for a broken query. The pre-filter
            # should make this unreachable; never record a fake near-zero.
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
        "model": model_key,
        "category": category,
        "backend": getattr(processor, "processor", None),
        "task": f"{TASK_QUERY_VERSION}-{TASK_DOC_VERSION}",
        "n_queries_requested": sample["n_queries_requested"],
        "n_queries_measured": len(per_query_seconds),
        "seed": sample["seed"],
        "query_idxs": measured_idxs,
        "standardized_batch_size": BATCH_SIZE,
        "scores_kwargs": scores_kwargs,
        "builder_batching": _BUILDER_BATCHING_NOTE.get(model_key),
        "per_query_seconds": per_query_seconds,
        "mean_seconds": float(arr.mean()) if len(arr) else None,
        "median_seconds": float(np.median(arr)) if len(arr) else None,
        "std_seconds": float(arr.std(ddof=1)) if len(arr) > 1 else None,
    }

    save_dir = Path(args.save_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{model_key}.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Sample the shared N_QUERIES query indices once and write "
        "them to <save-to>/query_sample.json, then exit - no --models "
        "needed. Run this exactly once before submitting any per-model "
        "timing jobs; every one of them reads this same file so every "
        "model is timed on the identical set of queries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="KEY",
        default=None,
        help="Models to time (default: all of them). Each is timed on its "
        "PRODUCTION backend, on the same shared query sample.",
    )
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

    if args.n_queries != N_QUERIES or args.seed != SEED:
        print(
            f"[!!] --n-queries/--seed are ignored for a timing run (got "
            f"n_queries={args.n_queries}, seed={args.seed}) - every model "
            f"reads the shared query_sample.json instead. Re-run "
            f"--generate-sample first if you actually want different "
            f"values, then re-run every timing job so they all still "
            f"agree on the same sample."
        )

    print("[~] Loading dataset...")
    queries, documents = load_data()

    # One shared query sample, so a latency table never compares models across
    # different query sets.
    sample = _load_query_sample(args.save_to)
    query_idxs = sample["query_idxs"]
    print(
        f"[~] Loaded {len(query_idxs)} shared query indices from "
        f"query_sample.json (seed={sample['seed']})"
    )

    models = args.models or sorted(ALL_BUILDERS)
    unknown = [m for m in models if m not in ALL_BUILDERS]
    if unknown:
        raise SystemExit(
            f"No timing builder for: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(ALL_BUILDERS))}"
        )

    failures = []
    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(f"# timing: {model_key} ({i}/{len(models)})")
        print("#" * 60)
        try:
            time_one_model(model_key, args, queries, documents, sample, query_idxs)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[!!] {model_key} failed: {e}")
            failures.append(model_key)

    print("\n" + "=" * 60)
    print(f"Done. {len(models) - len(failures)}/{len(models)} succeeded.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
