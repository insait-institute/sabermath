import os
import sys
from pathlib import Path

# Set CUDA devices strictly if needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# math-vs-word has its own environment.yml, separate from the main
# sabermath package's - so `import sabermath` isn't guaranteed to resolve
# without this, same convention as scripts/measure_query_time.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
# Reuses scripts/run_rerankers.py's own verified builders for the RaDeR
# bi-encoders and INF-Retriever-v1-Pro instead of duplicating their
# hard-won recipes here - see each one's own docstring in that file for the
# full history (chat-template corruption, wedged vLLM engines, etc.).
# Importable safely: everything below its `if __name__ == "__main__":`
# guard, no argparse/model-loading side effects at import time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from run_rerankers import (  # noqa: E402
    RADER_BIENCODER_MODELS,
    _build_inf_retriever_processor,
    _build_rader_biencoder_vllm,
)
from sabermath.processors import (  # noqa: E402
    ColBERTProcessor,
    GoogleProcessor,
    GroupRankProcessor,
    Qwen3RerankerVLLMProcessor,
    RaDeRRerankerVLLMProcessor,
    Rank1Processor,
    SentenceTransformersProcessor,
    SpladeProcessor,
    VLLMProcessor,
)

# Every model below that ALSO appears in scripts/run_rerankers.py's
# GENERIC_MODELS (the actual harness that produced the paper's reported
# numbers) is loaded exactly the way that entry specifies. qwen3-embedding-8b
# is the only overlap: {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True}
# with no init_kwargs - i.e. plain VLLMProcessor.from_huggingface(name), no
# pooler_config override (vLLM's own default pooling is trusted there).
#
# No other model in ALLOWED_MODELS below appears in GENERIC_MODELS at all, so
# for all of them this falls through to sabermath.benchmark._make_processor's
# own default (use_vllm=False) - i.e. plain
# SentenceTransformersProcessor.from_huggingface(name, trust_remote_code=True).
# That is the one general-purpose HuggingFace loading path this repo's
# processors actually define - EXCEPT the 4 models in _CUSTOM_BUILDERS below,
# confirmed (test_model_loading.py, jobs run 2026-08-22) to crash under that
# generic path with no workaround:
#   - harrier-oss-v1-27b / KaLM-Embedding-Gemma3-12B-2511: both Gemma3-based;
#     sentence-transformers' Transformer module unconditionally calls
#     AutoProcessor.from_pretrained(), which routes Gemma3 to a multimodal
#     image-processor loader neither (text-only) repo ships -
#     OSError: Can't load image processor for '<repo>'.
#   - Octen-Embedding-4B/8B: both ship a 2_Normalize/config.json with the old
#     {"normalize_embeddings": true} kwarg, but sentence-transformers'
#     current Normalize module takes no constructor args at all -
#     TypeError: Normalize.__init__() got an unexpected keyword argument
#     'normalize_embeddings'.
# The two builders below are ported verbatim from
# scripts/measure_query_time.py's _build_gemma3_text_embedding_processor()/
# _build_octen_processor() - the only validated recipes for these 4 models
# in this repo. pooling_mode="lasttoken" and the trailing L2-normalize are
# not a guess - both fetched directly from each repo's own
# 1_Pooling/config.json (pooling_mode_lasttoken: true) / modules.json
# (a 2_Normalize module).
GOOGLE_MODELS = {"google/gemini-embedding-001", "google/gemini-embedding-2"}

# The Gemini calls originally went through an internal AI-gateway proxy
# instead of hitting Google directly - dropped (2026-08-22) after
# confirming, independent of this code, that the gateway itself rejects
# every key tried against it (curl -H "x-goog-api-key: <key>"
# .../v1beta/models -> {"error":"Invalid Key"}), while the SAME key
# succeeds immediately against Google's real endpoint
# (generativelanguage.googleapis.com). Matches
# scripts/measure_query_time.py's own CLOSED_API_BUILDERS convention too
# (GoogleProcessor(name), no base_url) - going direct is both the thing
# proven to work and the thing that matches production.


def _build_gemma3_text_embedding_processor(model_name: str):
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
                "_RawTextTransformer is a load_models.py workaround only, "
                "not meant to be saved."
            )

    transformer = _RawTextTransformer()
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(), pooling_mode="lasttoken"
    )
    normalize = modules.Normalize()
    st = SentenceTransformer(modules=[transformer, pooling, normalize])
    return SentenceTransformersProcessor(st, model_name)


def _build_octen_processor(model_name: str):
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
    return SentenceTransformersProcessor(st, model_name)


# text-embedding-3-large/small via OpenRouter, not OpenAI directly - reuses
# the existing .openroutertok key (already used by
# scripts/measure_query_time.py's OpenRouterEmbeddingProcessor for these
# exact two models) rather than requiring a separate real OpenAI key.
# sabermath.processors.OpenAIProcessor has no way to point at a custom
# base_url, so this replicates its async batching/retry logic here instead
# of touching that file - ported verbatim from measure_query_time.py's own
# class of the same name.
class OpenRouterEmbeddingProcessor:
    processor = "openrouter"

    def __init__(self, model_name: str, api_key: str | None = None):
        import os as _os
        import warnings as _warnings

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Please install openai to use it as a processor") from e

        client_args = {"base_url": "https://openrouter.ai/api/v1"}

        if api_key is not None:
            client_args["api_key"] = api_key
        elif _os.getenv("OPENROUTER_API_KEY"):
            client_args["api_key"] = _os.getenv("OPENROUTER_API_KEY")
        else:
            _warnings.warn(
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
        import asyncio as _asyncio

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
                    await _asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"Failed to encode text after {retries} attempts."
        ) from last_error

    async def encode_async(
        self,
        texts,
        show_progress_bar: bool = True,
        *,
        max_concurrency: int = 20,
        retries: int = 3,
        **kwargs,
    ):
        import asyncio as _asyncio

        import numpy as _np

        if max_concurrency <= 0:
            raise ValueError('"max_concurrency" must be >= 1')
        sem = _asyncio.Semaphore(max_concurrency)
        coros = [self._encode_one(text, sem, retries=retries, **kwargs) for text in texts]
        results = await _asyncio.gather(*coros)
        return _np.asarray(results, dtype=_np.float32)

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        max_concurrency: int = 20,
        retries: int = 3,
        **kwargs,
    ):
        # No caching/get_scores in the original class - added here so this
        # matches every other processor's uniform get_scores() interface
        # that sim_embeddings.py calls generically.
        import asyncio as _asyncio

        import numpy as _np

        async def _run():
            embeddings = await self.encode_async(
                documents + [query],
                show_progress_bar=show_progress_bar,
                max_concurrency=max_concurrency,
                retries=retries,
                **kwargs,
            )
            query_emb = embeddings[-1]
            doc_embs = embeddings[:-1]
            q_norm = _np.linalg.norm(query_emb)
            d_norms = _np.linalg.norm(doc_embs, axis=1, keepdims=True)
            return (doc_embs / d_norms) @ (query_emb / q_norm)

        return _asyncio.run(_run())


# jina-v5-nano/-small's custom remote-code modeling forward() hard-requires
# a task to be set - NOT optional/stylistic (confirmed on job 736681:
# "ValueError: Task must be specified before encoding data" - it doesn't
# default to anything on its own). Set at load time via model_kwargs
# (equivalent to the old embed.py's per-call task="retrieval" encode kwarg,
# which no longer exists now that encoding goes through the generic
# EmbeddingProcessor.get_scores() -> encode() path with no per-model hook).
_EXTRA_MODEL_KWARGS = {
    "jinaai/jina-embeddings-v5-text-nano": {"default_task": "retrieval"},
    "jinaai/jina-embeddings-v5-text-small": {"default_task": "retrieval"},
}

# RaDeR bi-encoders' production protocol chunks long documents rather than
# silently truncating - ported from
# scripts/run_rerankers.py's _RADER_BIENCODER_SCORES_KWARGS. Only 0.32% of
# documents exceed 2048 tokens, but the numbers must stay
# preprocessing-identical to the paper's own protocol.
_RADER_SCORES_KWARGS = {"chunk_to_context": True, "context_length": 2048}

_SCORES_KWARGS = {
    RADER_BIENCODER_MODELS["rader-14b"]: _RADER_SCORES_KWARGS,
    RADER_BIENCODER_MODELS["rader-7b"]: _RADER_SCORES_KWARGS,
    RADER_BIENCODER_MODELS["rader-3b"]: _RADER_SCORES_KWARGS,
}


def get_scores_kwargs(model_id: str) -> dict:
    """Extra kwargs sim_embeddings.py must forward to processor.get_scores()
    for this specific model - empty for every model except the RaDeR
    bi-encoder family (see _SCORES_KWARGS above)."""
    return dict(_SCORES_KWARGS.get(model_id, {}))


# Every entry below that also appears in scripts/run_rerankers.py's
# CUSTOM_MODEL_BUILDERS or GENERIC_MODELS (the harness that produced the
# paper's reported numbers) is built via the EXACT same call - pooler
# configs, hf_overrides, dtype pins, and all - since these are load-bearing
# per that file's own extensive comments (silently wrong pooling has
# corrupted numbers here before, e.g. the RaDeR mean-pooling incident).
# Keyed by the model's own HF repo string where one naturally exists (EMBED
# entries); by the short key run_rerankers.py itself uses where it doesn't
# (RERANK entries built from multi-argument processor constructors, not a
# bare "give me this HF repo" call).
_CUSTOM_BUILDERS = {
    # --- Gemma3/Octen loading-crash workarounds (unchanged) ---
    "microsoft/harrier-oss-v1-27b": _build_gemma3_text_embedding_processor,
    "tencent/KaLM-Embedding-Gemma3-12B-2511": _build_gemma3_text_embedding_processor,
    "Octen/Octen-Embedding-4B": _build_octen_processor,
    "Octen/Octen-Embedding-8B": _build_octen_processor,
    # --- EMBED: GENERIC_MODELS-equivalent vLLM builds (pooler_config
    # load-bearing - vLLM's own arch default is NOT trusted for these) ---
    "Qwen/Qwen3-Embedding-8B": lambda name: VLLMProcessor.from_huggingface(name),
    "hanhainebula/reason-embed-qwen3-8b-0928": lambda name: VLLMProcessor.from_huggingface(
        name, pooler_config={"pooling_type": "LAST", "normalize": True}
    ),
    "AQ-MedAI/Diver-Retriever-4B": lambda name: VLLMProcessor.from_huggingface(
        name, pooler_config={"pooling_type": "LAST", "normalize": True}
    ),
    "AQ-MedAI/Diver-Retriever-0.6B": lambda name: VLLMProcessor.from_huggingface(
        name,
        pooler_config={"pooling_type": "LAST", "normalize": True},
        dtype="bfloat16",
    ),
    # --- EMBED: CUSTOM_MODEL_BUILDERS-equivalent vLLM builds ---
    "reasonir/ReasonIR-8B": lambda name: VLLMProcessor.from_huggingface(
        name,
        hf_overrides={
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        pooler_config={"pooling_type": "MEAN", "normalize": True},
    ),
    RADER_BIENCODER_MODELS["rader-14b"]: lambda name: _build_rader_biencoder_vllm(name),
    RADER_BIENCODER_MODELS["rader-7b"]: lambda name: _build_rader_biencoder_vllm(name),
    RADER_BIENCODER_MODELS["rader-3b"]: lambda name: _build_rader_biencoder_vllm(name),
    "infly/inf-retriever-v1-pro": lambda name: _build_inf_retriever_processor(name),
    # --- RERANK: short-keyed (no single natural HF-repo dispatch) ---
    "jhu-clsp/rank1-32b": lambda name: Rank1Processor(tensor_parallel_size=1),
    "qwen3-reranker-8b": lambda name: Qwen3RerankerVLLMProcessor(),
    "qwen3-reranker-4b": lambda name: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-4B"
    ),
    "qwen3-reranker-0.6b": lambda name: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-0.6B"
    ),
    "splade-code-8b": lambda name: SpladeProcessor(),
    "splade-code-0.6b": lambda name: SpladeProcessor("naver/splade-code-06B"),
    "rader-reranker-7b": lambda name: RaDeRRerankerVLLMProcessor(),
    "diver-grouprank-32b": lambda name: GroupRankProcessor(tensor_parallel_size=1),
    # --- RERANK: ColBERT (natural HF-repo dispatch) ---
    "lightonai/GTE-ModernColBERT-v1": lambda name: ColBERTProcessor(name),
    "lightonai/Reason-ModernColBERT": lambda name: ColBERTProcessor(name),
}

# text-embedding-3-large/small: OpenRouter, not the generic ST/vLLM path at
# all - see OpenRouterEmbeddingProcessor above.
_OPENROUTER_MODELS = {"text-embedding-3-large", "text-embedding-3-small"}


def get_model(MODEL_ID: str):
    if MODEL_ID in GOOGLE_MODELS:
        api_model_name = MODEL_ID.removeprefix("google/")
        print(f"Loading {MODEL_ID} via GoogleProcessor ({api_model_name})...")
        return GoogleProcessor(api_model_name)

    if MODEL_ID in _OPENROUTER_MODELS:
        print(f"Loading {MODEL_ID} via OpenRouterEmbeddingProcessor...")
        return OpenRouterEmbeddingProcessor(MODEL_ID)

    if MODEL_ID in _CUSTOM_BUILDERS:
        print(f"Loading {MODEL_ID} via its custom builder (see _CUSTOM_BUILDERS)...")
        return _CUSTOM_BUILDERS[MODEL_ID](MODEL_ID)

    print(f"Loading {MODEL_ID} via SentenceTransformersProcessor.from_huggingface...")
    extra_model_kwargs = _EXTRA_MODEL_KWARGS.get(MODEL_ID)
    return SentenceTransformersProcessor.from_huggingface(
        MODEL_ID,
        trust_remote_code=True,
        **({"model_kwargs": extra_model_kwargs} if extra_model_kwargs else {}),
    )


ALLOWED_MODELS = [
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "google/embeddinggemma-300m",
    "google-bert/bert-base-uncased",  # Standard BERT
    "FacebookAI/roberta-base",  # Standard RoBERTa
    "microsoft/codebert-base",
    "google/gemini-embedding-001",
    "google/gemini-embedding-2",
    "microsoft/harrier-oss-v1-0.6b",
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-27b",
    "Octen/Octen-Embedding-4B",
    "Octen/Octen-Embedding-8B",
    "jinaai/jina-embeddings-v5-text-nano",
]

# Added 2026-08-22 to cover the rest of the paper's model table beyond the
# 17 above (which were the original math-vs-word roster). See load_models.py's
# _CUSTOM_BUILDERS/get_scores_kwargs for how each of these is actually
# built/scored - many need a specific pooler_config/dtype/chunking that
# differs from the generic SentenceTransformersProcessor.from_huggingface()
# fallback the 17 above mostly use.
ADDITIONAL_MODELS = [
    # EMBED
    "hanhainebula/reason-embed-qwen3-8b-0928",  # Reason-Embed-Qwen3-8B
    "AQ-MedAI/Diver-Retriever-4B",
    "AQ-MedAI/Diver-Retriever-0.6B",
    "infly/inf-retriever-v1-pro",  # INF-Retriever-v1-Pro
    "reasonir/ReasonIR-8B",  # ReasonIR-8B
    RADER_BIENCODER_MODELS["rader-14b"],
    RADER_BIENCODER_MODELS["rader-7b"],
    RADER_BIENCODER_MODELS["rader-3b"],
    "nvidia/llama-embed-nemotron-8b",  # LLaMA-Embed-Nemotron-8B
    "intfloat/multilingual-e5-large",  # Multilingual-E5-Large
    "jinaai/jina-embeddings-v5-text-small",  # Jina-v5-Text-Small
    # API (OpenRouter - see OpenRouterEmbeddingProcessor)
    "text-embedding-3-large",
    "text-embedding-3-small",
    # RERANK
    "jhu-clsp/rank1-32b",  # Rank1-32B
    "qwen3-reranker-8b",
    "qwen3-reranker-4b",
    "qwen3-reranker-0.6b",
    "splade-code-8b",
    "splade-code-0.6b",
    "rader-reranker-7b",  # RaDeR-Reranker-7B
    "diver-grouprank-32b",  # Diver-GroupRank-32B
    "lightonai/GTE-ModernColBERT-v1",  # GTE-ModernColBERT
    "lightonai/Reason-ModernColBERT",  # Reason-ModernColBERT
]

ALLOWED_MODELS = ALLOWED_MODELS + ADDITIONAL_MODELS
