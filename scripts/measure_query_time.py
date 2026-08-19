"""Measure standardized average per-query task time across every model in
the SaberMath benchmark's results tables (rerankers, bi-encoder embedding
models - open and closed/API - and classical/lexical baselines).

Does NOT modify any existing processor or run_rerankers.py - this is a
separate, standalone measurement harness with its own methodology, agreed
on separately from the correctness-focused nDCG runs:

  1. Rerankers are run normally - their own native batching is untouched
     (e.g. rank1-32b's one batched vllm.generate() call over all
     candidates, qwen3-reranker-8b's batch_size=8, splade's batch_size=4).

  2. Embedding models (bi-encoder, open HF or closed/API) have their
     amortization effect removed: every one of the N_QUERIES sampled
     queries is scored from a cold cache - no candidate document
     embedding is precomputed or reused across queries - via
     EmbeddingProcessor's own existing check_cache=False/update_cache=False
     kwargs (no processor code changes needed, this flag already existed).

  3. Those same embedding models are forced to batch_size=16 wherever
     parallelization is meaningful, standardizing the throughput
     assumption across models that would otherwise each use their own
     tuned default. Two documented exceptions, not silent overrides:
       - OpenAIProcessor has no batch_size concept at all (one HTTP
         request per document, concurrency-limited) - max_concurrency=16
         is used instead as the closest analogous "parallelism fixed at
         16" knob.
       - GoogleProcessor already hardcodes batch_size=1 internally for
         gemini-embedding-2 specifically (existing code, not touched
         here) - batch_size=16 is still passed for consistency, but that
         model-specific override downgrades it regardless.

  4. Everything is meant to run on a single H200 GPU node (see
     scripts/timing/run_timing.slurm) - including the classical/closed-API
     models, which don't need the GPU themselves, for node/environment
     consistency.

  5. Model loading is excluded from the timed region. Only the time from
     "start scoring this query's candidates" to "have the final sorted
     ranking" is measured, per query.

Classical/lexical methods (bm25, tf-idf, jaccard, approach0) have no
caching or batching concept at all - confirmed by reading each processor
directly: TfidfProcessor refits its vectorizer fresh on every call,
BM25Processor/Approach0Processor rebuild their index fresh on every call.
They're just called plainly - "whenever parallelization is possible"
correctly excludes them.

No warmup query - every one of the N_QUERIES sampled queries is timed,
including the first (deliberately, per instruction - a model whose first
call pays a one-time JIT/CUDA-graph-capture cost will show that in its
per-query numbers, not have it silently discarded).

Task is fixed to statement-full (query="statement" version, i.e. the bare
"problem" field; documents="full" version, i.e. "Problem: ...\n\nSolution:
...") - the paper's main setting, matching what the nDCG results already
reported use.

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
    Qwen3RerankerProcessor,
    Rank1Processor,
    ReasonIRProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    TfidfProcessor,
)

# Reuses the already-verified lasttoken-pooling fix instead of duplicating
# it here - see run_rerankers.py's own _build_rader_14b_processor docstring
# for why rader-14b specifically needs this (its HF repo ships no
# sentence-transformers module config to auto-detect at all, which
# silently produced the wrong pooling mode when loaded generically).
from run_rerankers import _build_rader_14b_processor  # noqa: E402

# --- how many queries to average over per model. Also overridable via
# --n-queries. ---
N_QUERIES = 50

TASK_QUERY_VERSION = "statement"
TASK_DOC_VERSION = "full"
SEED = 42
BATCH_SIZE = 16  # standardized encode-call batch size for embedding models


class OpenRouterEmbeddingProcessor(EmbeddingProcessor):
    """text-embedding-3-small/large via OpenRouter instead of OpenAI
    directly - requested explicitly (OpenRouter key supplied instead of an
    OpenAI one). sabermath.processors.OpenAIProcessor has no way to point
    at a custom base_url, and per this harness's own "don't modify
    existing processors" rule, this replicates its exact async
    batching/retry logic here instead of touching that file, just with a
    different base_url and the OpenRouter model-id convention
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


def _st(model_name: str, extra_model_kwargs: dict | None = None):
    """Plain bi-encoder via the generic SentenceTransformersProcessor path,
    same trust_remote_code=True/model_kwargs={"torch_dtype": "auto"}
    convention already used for reason-embed-qwen3-8b/diver-retriever-4b
    in run_rerankers.py's GENERIC_MODELS. extra_model_kwargs merges in on
    top of that default (used by the jina-embeddings-v5-* entries below,
    which additionally need default_task set)."""
    model_kwargs = {"torch_dtype": "auto"}
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)
    return STProcessor.from_huggingface(
        model_name,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )


def _build_gemma3_text_embedding_processor(model_name: str):
    """microsoft/harrier-oss-v1-27b and tencent/KaLM-Embedding-Gemma3-12B-2511:
    both Gemma3-based text-only bi-encoders. Loading either through ANY path
    that constructs sentence_transformers' own
    sentence_transformers.sentence_transformer.modules.Transformer -
    including the plain SentenceTransformer(model_name) auto-load AND the
    rader-14b-style explicit modules.Transformer(model_name) construction -
    crashes, because that class unconditionally calls
    AutoProcessor.from_pretrained() in __init__ (sentence-transformers==5.7.0,
    base/modules/transformer.py L671 - no parameter skips it). transformers
    routes gemma3's model_type to a multimodal Processor mapping that then
    tries to load an image processor config neither text-only repo ships:
        OSError: Can't load image processor for '<repo>' ...
    Confirmed directly: a bare AutoProcessor.from_pretrained() call on both
    repos reproduces this exact error, while AutoTokenizer.from_pretrained()
    on the same repos works fine (it has no image-processor path at all) -
    which is all a text-only bi-encoder actually needs. _RawTextTransformer
    below is a minimal drop-in that only ever touches AutoTokenizer/
    AutoModel, sidestepping AutoProcessor (and this whole crash) entirely.

    pooling_mode="lasttoken" and the trailing L2-normalize are not a guess -
    both repos' own 1_Pooling/config.json (pooling_mode_lasttoken: true) and
    modules.json (a 2_Normalize module) were fetched and read directly
    rather than assumed from the model cards.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules
    from transformers import AutoModel, AutoTokenizer

    class _RawTextTransformer(modules.InputModule):
        save_in_root = True

        def __init__(self):
            super().__init__()
            self._auto_model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        def preprocess(self, inputs, prompt=None, **kwargs):
            if prompt:
                inputs = self._prepend_prompt(inputs, prompt)
            return dict(
                self.tokenizer(
                    list(inputs), padding=True, truncation=True, return_tensors="pt"
                )
            )

        def forward(self, features, **kwargs):
            model_inputs = {
                k: v
                for k, v in features.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            outputs = self._auto_model(**model_inputs)
            features["token_embeddings"] = outputs.last_hidden_state
            return features

        def get_embedding_dimension(self) -> int:
            return self._auto_model.config.hidden_size

        def save(self, output_path, *args, **kwargs):
            raise NotImplementedError(
                "_RawTextTransformer is timing-harness-only and not meant to be saved."
            )

    transformer = _RawTextTransformer()
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(), pooling_mode="lasttoken"
    )
    normalize = modules.Normalize()
    st = SentenceTransformer(modules=[transformer, pooling, normalize])
    return STProcessor(st, model_name)


def _build_octen_processor(model_name: str):
    """Octen/Octen-Embedding-4B and -8B: sentence-transformers==5.7.0's
    Normalize module (sentence_transformer/modules/normalize.py) no longer
    accepts ANY constructor argument - it just always L2-normalizes
    unconditionally now. Both Octen repos still ship a
    2_Normalize/config.json with the old {"normalize_embeddings": true}
    kwarg (confirmed directly by fetching each repo's own file), which the
    generic auto-loading path tries to pass into Normalize(**config) and
    crashes with a TypeError. Unlike harrier-oss-v1-27b/
    kalm-embedding-gemma3-12b-2511 above, Octen's own Transformer module
    loads fine on its own (these are Qwen-based, not Gemma3 - no
    AutoProcessor image-processor issue), so this only needs to skip
    loading that one broken Normalize config, not bypass AutoProcessor.
    pooling_mode="lasttoken" per each repo's own 1_Pooling/config.json
    (pooling_mode_lasttoken: true), fetched directly rather than assumed.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    transformer = modules.Transformer(
        model_name,
        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto"},
        config_kwargs={"trust_remote_code": True},
    )
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(), pooling_mode="lasttoken"
    )
    normalize = modules.Normalize()
    st = SentenceTransformer(modules=[transformer, pooling, normalize])
    return STProcessor(st, model_name)


RERANKER_BUILDERS = {
    "rank1-32b": lambda: Rank1Processor(tensor_parallel_size=1),
    "qwen3-reranker-8b": lambda: Qwen3RerankerProcessor(),
    "splade-code-8b": lambda: SpladeProcessor(),
    "gte-moderncolbert": lambda: ColBERTProcessor("lightonai/GTE-ModernColBERT-v1"),
    "reason-moderncolbert": lambda: ColBERTProcessor("lightonai/Reason-ModernColBERT"),
}

EMBEDDING_BUILDERS = {
    # Our own already-integrated bi-encoders.
    "reasonir-8b": lambda: ReasonIRProcessor(),
    "reason-embed-qwen3-8b": lambda: _st("hanhainebula/reason-embed-qwen3-8b-0928"),
    "diver-retriever-4b": lambda: _st("AQ-MedAI/Diver-Retriever-4B"),
    "rader-14b": lambda: _build_rader_14b_processor(),
    # qwen3-embedding-8b: deliberately uses the plain SentenceTransformer
    # path here, NOT VLLMProcessor - unlike the production reranker
    # pipeline (which uses vLLM for throughput on the full 1000-query
    # run). vLLM manages its own internal batching/scheduling; it doesn't
    # expose a client-side batch_size the way ST's .encode(batch_size=...)
    # does, and standardizing batch_size=16 across every embedding model
    # is the entire point of this harness. This is a deliberate,
    # documented deviation from the production model choice, not an
    # oversight.
    "qwen3-embedding-8b": lambda: _st("Qwen/Qwen3-Embedding-8B"),
    "qwen3-embedding-4b": lambda: _st("Qwen/Qwen3-Embedding-4B"),
    "qwen3-embedding-0.6b": lambda: _st("Qwen/Qwen3-Embedding-0.6B"),
    "harrier-oss-v1-270m": lambda: _st("microsoft/harrier-oss-v1-270m"),
    "harrier-oss-v1-0.6b": lambda: _st("microsoft/harrier-oss-v1-0.6b"),
    "harrier-oss-v1-27b": lambda: _build_gemma3_text_embedding_processor(
        "microsoft/harrier-oss-v1-27b"
    ),
    "bge-m3": lambda: _st("BAAI/bge-m3"),
    "llama-embed-nemotron-8b": lambda: _st("nvidia/llama-embed-nemotron-8b"),
    "kalm-embedding-gemma3-12b-2511": lambda: _build_gemma3_text_embedding_processor(
        "tencent/KaLM-Embedding-Gemma3-12B-2511"
    ),
    "roberta-base": lambda: _st("FacebookAI/roberta-base"),
    "bert-base-uncased": lambda: _st("google-bert/bert-base-uncased"),
    "embeddinggemma-300m": lambda: _st("google/embeddinggemma-300m"),
    "multilingual-e5-large": lambda: _st("intfloat/multilingual-e5-large"),
    "jina-embeddings-v5-text-nano": lambda: _st(
        "jinaai/jina-embeddings-v5-text-nano", {"default_task": "retrieval"}
    ),
    "jina-embeddings-v5-text-small": lambda: _st(
        "jinaai/jina-embeddings-v5-text-small", {"default_task": "retrieval"}
    ),
    "octen-embedding-4b": lambda: _build_octen_processor("Octen/Octen-Embedding-4B"),
    "octen-embedding-8b": lambda: _build_octen_processor("Octen/Octen-Embedding-8B"),
}

# Closed/API embedding models. NOTE on why these build GoogleProcessor/
# OpenAIProcessor directly rather than going through a bare model-name
# string: experiments/confidence_intervals/confidence.py lists these exact
# strings (e.g. "gemini-embedding-001") in its own cached_models and calls
# sabermath.evaluate(model, use_vllm=use_vllm, ...) on them - but grepping
# that file shows `use_vllm` is never actually defined anywhere in it (it
# would raise NameError if run), and even if it were, a bare string only
# ever resolves to STProcessor/VLLMProcessor in benchmark.py's
# _make_processor() - neither of which is correct for an API model
# identifier that isn't an HF repo path. So this harness builds the right
# processor directly instead of relying on that apparently-nonfunctional
# auto-dispatch.
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


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _get_scores_kwargs(category: str, model_key: str) -> dict:
    if category in ("reranker", "classical"):
        return {}
    if category == "embedding":
        return {"check_cache": False, "update_cache": False, "batch_size": BATCH_SIZE}
    if category == "closed_api":
        if model_key.startswith("text-embedding"):
            return {
                "check_cache": False,
                "update_cache": False,
                "max_concurrency": BATCH_SIZE,
            }
        return {"check_cache": False, "update_cache": False, "batch_size": BATCH_SIZE}
    raise ValueError(f"unknown category {category!r}")


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
        "task": f"{TASK_QUERY_VERSION}-{TASK_DOC_VERSION}",
        "n_queries_requested": sample["n_queries_requested"],
        "n_queries_measured": len(per_query_seconds),
        "seed": sample["seed"],
        "query_idxs": measured_idxs,
        "scores_kwargs": scores_kwargs,
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
