from .base import ModelProcessor
from .embedding_processor import EmbeddingProcessor
from .st_processor import SentenceTransformersProcessor
from .vllm_processor import VLLMProcessor
from .unknown_processor import UnknownProcessor

from .google_processor import GoogleProcessor
from .openai_processor import OpenAIProcessor
from .openrouter_processor import OpenRouterEmbeddingProcessor

# The lexical baselines need the optional "lexical" extra. Imported
# defensively so a plain `pip install -e .` still yields a working package:
# only instantiating one of them without the extra raises, rather than the
# import breaking `sabermath` for every other model too.
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
from .colbert_processor import ColBERTProcessor
from .reasonir_processor import ReasonIRProcessor
from .splade_processor import SpladeProcessor
from .grouprank_processor import GroupRankProcessor

# The two cross-encoder reranker families. Heavy deps (vllm, peft) are
# imported lazily inside _init, so these imports stay cheap.
from .qwen3_reranker_vllm_processor import Qwen3RerankerVLLMProcessor
from .rader_reranker_vllm_processor import RaDeRRerankerVLLMProcessor

# infly's composed query-aligner + dense-retriever system, and the shared ST
# load recipe the standalone inf-retriever-v1-pro entry uses.
from .infx_retriever_processor import INFXRetrieverProcessor, build_inf_retriever_st

# A generative pointwise reranker, scored from a <score> tag the model writes
# rather than from logprobs.
from .retro_star_processor import RetroStarProcessor, RetroStarRewrittenProcessor

# The BGE-Reasoner "rewrite, then embed" system as one entry.
from .reason_rewriter_processor import ReasonRewriterProcessor
