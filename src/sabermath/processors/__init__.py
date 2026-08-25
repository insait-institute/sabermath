from .base import ModelProcessor
from .embedding_processor import EmbeddingProcessor
from .st_processor import SentenceTransformersProcessor
from .vllm_processor import VLLMProcessor
from .unknown_processor import UnknownProcessor

from .google_processor import GoogleProcessor
from .openai_processor import OpenAIProcessor
from .openrouter_processor import OpenRouterEmbeddingProcessor

# The legacy/lexical baselines (BM25, TF-IDF, Jaccard) depend on the
# optional "legacy" extra (rank_bm25, scikit-learn, pya0 - see
# pyproject.toml's [project.optional-dependencies]). TfidfProcessor imports
# sklearn directly, and all three also import pya0 transitively via
# tokenization_helper. Import them defensively so a plain `pip install -e .`
# (no extras - what every other model here actually needs) still gets a
# working `sabermath` package; only actually instantiating one of these
# three without the extra installed raises (via TypeError on `None(...)`,
# since that's a clearer failure point than an import-time crash for
# everyone else). Confirmed the hard way: this used to break importing
# `sabermath` at all for every reranker model, not just the legacy ones.
try:
    from .tf_idf_processor import TfidfProcessor
except ImportError:
    TfidfProcessor = None
try:
    from .jaccard_processor import JaccardProcessor
except ImportError:
    JaccardProcessor = None
try:
    from .bm25_processor import BM25Processor
except ImportError:
    BM25Processor = None

from .approach0_processor import Approach0Processor

from .rank1_processor import Rank1Processor
from .qwen3_reranker_processor import Qwen3RerankerProcessor
from .colbert_processor import ColBERTProcessor
from .reasonir_processor import ReasonIRProcessor
from .splade_processor import SpladeProcessor
from .grouprank_processor import GroupRankProcessor
from .rader_reranker_processor import RaDeRRerankerProcessor

# vLLM-backed production defaults (since 2026-08-20) for the two cross-encoder
# reranker families above; the HF-transformers versions stay as the legacy
# reference paths (scripts/test_vllm_feasibility.py compares the two). Heavy
# deps (vllm, peft) are imported lazily inside _init, so these imports are as
# cheap as the rest.
from .qwen3_reranker_vllm_processor import Qwen3RerankerVLLMProcessor
from .rader_reranker_vllm_processor import RaDeRRerankerVLLMProcessor

# INF-X-Retriever: infly's composed query-aligner + dense-retriever system
# (and the shared, validated ST load recipe for the standalone
# inf-retriever-v1-pro entry). Heavy deps imported lazily at construction.
from .infx_retriever_processor import INFXRetrieverProcessor, build_inf_retriever_st
