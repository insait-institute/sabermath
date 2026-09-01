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
# import breaking `sabermath` for every other model too. The stand-in below
# raises with the missing package and the install command, so the failure is
# never a bare "'NoneType' object is not callable" at the call site.
# pya0 is a base dependency (Linux-only wheel), the rest come from an extra,
# so the hint has to match what is actually missing.
_BASE_PACKAGES = {"pya0"}


def _unavailable(name: str, extra: str, cause: ImportError):
    missing = getattr(cause, "name", None) or str(cause)
    if missing in _BASE_PACKAGES:
        how = "pip install -e ."
        note = f" ({missing} ships as a Linux-only wheel)"
    else:
        how = f'pip install -e ".[{extra}]"'
        note = ""

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} is unavailable: {missing!r} is not installed{note}. "
                f"Install it with: {how}"
            ) from cause

    _Unavailable.__name__ = name
    return _Unavailable


try:
    from .tf_idf_processor import TfidfProcessor
except ImportError as e:
    TfidfProcessor = _unavailable("TfidfProcessor", "lexical", e)
try:
    from .jaccard_processor import JaccardProcessor
except ImportError as e:
    JaccardProcessor = _unavailable("JaccardProcessor", "lexical", e)
try:
    from .bm25_processor import BM25Processor
except ImportError as e:
    BM25Processor = _unavailable("BM25Processor", "lexical", e)

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
