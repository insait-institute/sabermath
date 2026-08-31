import os
import sys
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Kept so this module still imports when the package has not been pip-installed
# into the active environment - the same convention scripts/run_timing.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from sabermath.registry import RADER_BIENCODER_MODELS  # noqa: E402
from sabermath.processors import SentenceTransformersProcessor  # noqa: E402
from . import SIMILARITIES_DIR

# Nothing here maintains its own builders or per-model get_scores kwargs:
# every model is built and scored by delegating to the registry through
# model_key_for(), so "how is this model called" has one answer per model.
# microsoft/codebert-base is the only exception - it is not in the paper's
# table and has no registry key, so it keeps the generic ST path.
_NO_EXPERIMENT_KEY = {"microsoft/codebert-base"}

# tensor_parallel_size. This module pins CUDA_VISIBLE_DEVICES=0 above, so
# there is exactly one visible GPU and tp>1 is not expressible. One
# math-vs-word method runs per invocation, on one GPU.
_TENSOR_PARALLEL_SIZE = 1

# Only read by the registry's inf-x-retriever builder, which is not in
# ALLOWED_MODELS - but the processor builder takes it unconditionally.
_SAVE_DIR = str(SIMILARITIES_DIR)


# HF repo string -> the registry's short model key, built by inverting the
# registry's own spec dicts rather than hand-maintaining a second roster.
def _model_key_by_id() -> dict:
    from sabermath import registry as rr

    mapping = {}
    for spec_dict in (rr.GENERIC_MODELS, getattr(rr, "TABLE_MODELS", {})):
        for key, spec in spec_dict.items():
            if isinstance(spec, dict) and "model" in spec:
                mapping[spec["model"]] = key
    for key, repo in RADER_BIENCODER_MODELS.items():
        mapping[repo] = key
    for attr in ("QWEN3_RERANKER_REPOS", "COLBERT_REPOS"):
        for key, repo in getattr(rr, attr, {}).items():
            mapping[repo] = key
    for key, value in getattr(rr, "API_MODELS", {}).items():
        mapping.setdefault(value[1] if isinstance(value, tuple) else value, key)
    return mapping


def model_key_for(model_id: str) -> str:
    from sabermath import registry as rr

    if model_id in rr.ALL_MODEL_KEYS:
        return model_id
    key = _model_key_by_id().get(model_id)
    if key is None:
        # CUSTOM_MODEL_BUILDERS/API_MODELS entries have no "model" field to
        # invert, but their short key is the repo's last path component
        # lowercased. Accepted only when it resolves to a real key, so a model
        # genuinely outside the table still raises.
        candidate = model_id.rsplit("/", 1)[-1].lower()
        if candidate in rr.ALL_MODEL_KEYS:
            return candidate
    if key is None:
        raise KeyError(
            f"{model_id} has no scripts/run_experiments.py model key, so its "
            "input envelope and instruction handling are unknown. Add it to "
            "that file's spec dicts rather than special-casing it here."
        )
    return key


def _instruction_args(instruction_key: str | None) -> tuple[str, str | None]:
    from sabermath.instructions import INSTRUCTIONS

    if instruction_key is None:
        return "p0", INSTRUCTIONS["p0"]
    if instruction_key not in INSTRUCTIONS:
        raise ValueError(
            f"Unknown instruction key {instruction_key!r} - valid: "
            f"{sorted(INSTRUCTIONS)}"
        )
    return instruction_key, INSTRUCTIONS[instruction_key]


def get_scores_kwargs(
    model_id: str,
    instruction_key: str | None = None,
) -> dict:
    if model_id in _NO_EXPERIMENT_KEY:
        return {}

    from sabermath import registry as rr

    key, instruction_text = _instruction_args(instruction_key)
    kwargs, _ = rr.prompt_scores_kwargs(
        model_key_for(model_id), instruction_text
    )
    return kwargs


def wraps_instruction(
    model_id: str,
    instruction_key: str | None = None,
) -> bool:
    if model_id in _NO_EXPERIMENT_KEY:
        return True

    from sabermath import registry as rr

    key, instruction_text = _instruction_args(instruction_key)
    model_key = model_key_for(model_id)
    _, wrap = rr.prompt_scores_kwargs(model_key, instruction_text)
    if model_key in rr.QWEN3_RERANKER_REPOS:
        return False
    return wrap


def assert_envelope_supported(model_id: str, processor, scores_kwargs: dict) -> None:
    if model_id in _NO_EXPERIMENT_KEY:
        return

    from sabermath import registry as rr

    rr._assert_envelope_supported(model_key_for(model_id), processor, scores_kwargs)


def get_model(
    MODEL_ID: str,
    instruction_key: str | None = None,
):
    if MODEL_ID in _NO_EXPERIMENT_KEY:
        print(
            f"Loading {MODEL_ID} via SentenceTransformersProcessor."
            f"from_huggingface (no the registry key - see _NO_EXPERIMENT_KEY)..."
        )
        return SentenceTransformersProcessor.from_huggingface(
            MODEL_ID, trust_remote_code=True
        )

    from sabermath import registry as rr

    key, _ = _instruction_args(instruction_key)
    model_key = model_key_for(MODEL_ID)
    protocol = "canonical"
    print(
        f"Loading {MODEL_ID} as the registry model key {model_key!r} "
        f"(arm {key}, {protocol} protocol) via "
        f"the registry's processor builder..."
    )
    return rr.build_processor(
        model_key, key, _TENSOR_PARALLEL_SIZE, _SAVE_DIR
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

# The rest of the paper's model table. Treated identically to the list above;
# how each is built and scored is decided by the registry, via model_key_for().
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
    # API (OpenRouter - sabermath.registry.API_MODELS routes these)
    "text-embedding-3-large",
    "text-embedding-3-small",
    # RERANK
    "jhu-clsp/rank1-32b",  # Rank1-32B
    "jhu-clsp/rank1-7b",  # Rank1-7B
    "jhu-clsp/rank1-0.5b",  # Rank1-0.5B
    "qwen3-reranker-8b",
    "qwen3-reranker-4b",
    "qwen3-reranker-0.6b",
    "splade-code-8b",
    "splade-code-0.6b",
    "rader-reranker-7b",  # RaDeR-Reranker-7B
    "diver-grouprank-32b",  # Diver-GroupRank-32B
    "lightonai/GTE-ModernColBERT-v1",  # GTE-ModernColBERT
    "lightonai/Reason-ModernColBERT",  # Reason-ModernColBERT
    "inf-x-retriever",  # INF-X-Retriever
    # The rewritten arm. Short-keyed because composed systems have no single
    # repo to dispatch on. These put the instruction on the REWRITER only - do
    # NOT substitute reason-rewriter-reason-embed-8b-instructed, which also
    # instructs the encoder and is a different experiment.
    "retro-star-32b",  # Retro*-Qwen3-32B
    "retro-star-32b-rewritten",  # Retro*-Qwen3-32B on the rewritten query
    "reason-rewriter-reason-embed-8b",  # ReasonEmbed-Qwen3-8B-rewritten
    "reason-rewriter-reason-embed-llama-3.1-8b",  # ReasonEmbed-Llama-3.1-8B-rewritten
    "hanhainebula/reason-embed-llama-3.1-8b-0928",  # ReasonEmbed-Llama-3.1-8B
]

ALLOWED_MODELS = ALLOWED_MODELS + ADDITIONAL_MODELS
