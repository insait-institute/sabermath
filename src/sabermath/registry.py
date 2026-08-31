from __future__ import annotations

from pathlib import Path
from typing import get_args

from .instructions import INSTRUCTIONS
from .processors import (
    Approach0Processor,
    BM25Processor,
    ColBERTProcessor,
    GoogleProcessor,
    GroupRankProcessor,
    INFXRetrieverProcessor,
    JaccardProcessor,
    OpenAIProcessor,
    OpenRouterEmbeddingProcessor,
    Qwen3RerankerVLLMProcessor,
    RaDeRRerankerVLLMProcessor,
    Rank1Processor,
    ReasonIRProcessor,
    ReasonRewriterProcessor,
    RetroStarProcessor,
    RetroStarRewrittenProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    TfidfProcessor,
    VLLMProcessor,
)
from .schemas import Branch, Task

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

EXPERIMENT_COLBERT_QUERY_LENGTH = 256
EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE = 1280

VENDOR_QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

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
    "diver-grouprank-32b": "fixed rubric template, no task slot",
    "rader-reranker-7b": "T10 query:/Query:/document: template, no instruction slot",
}
INSTRUCTION_CONTROL_MODELS = frozenset(INSTRUCTION_CONTROL_REASONS)

REASONIR_QUERY_INSTRUCTION_SLOT = "<|user|>\n{instruction}\n<|embed|>\n"

RADER_EXPECTED_EOS = "<|im_end|>"

RADER_BIENCODER_MODELS = {
    "rader-3b": "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "rader-7b": "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "rader-14b": "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes",
}


def _build_rader_biencoder_vllm(model_name: str):
    return VLLMProcessor.from_huggingface(
        model_name,
        pooler_config={"pooling_type": "LAST", "normalize": False},
        dtype="bfloat16",
        max_model_len=2080,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )


def _build_inf_retriever_processor(model_name: str = "infly/inf-retriever-v1-pro"):
    from sabermath.processors import build_inf_retriever_st

    return STProcessor(build_inf_retriever_st(model_name), model_name)


CUSTOM_MODEL_BUILDERS = {
    "rank1-32b": lambda tp: Rank1Processor(tensor_parallel_size=1),
    "rank1-7b": lambda tp: Rank1Processor("jhu-clsp/rank1-7b"),
    "rank1-0.5b": lambda tp: Rank1Processor("jhu-clsp/rank1-0.5b"),
    "rank1-32b-bf16": lambda tp: Rank1Processor(
        tensor_parallel_size=1, dtype="bfloat16"
    ),
    "qwen3-reranker-8b": lambda tp: Qwen3RerankerVLLMProcessor(),
    "qwen3-reranker-4b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-4B"
    ),
    "qwen3-reranker-0.6b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-0.6B"
    ),
    "gte-moderncolbert": lambda tp: ColBERTProcessor("lightonai/GTE-ModernColBERT-v1"),
    "reason-moderncolbert": lambda tp: ColBERTProcessor("lightonai/Reason-ModernColBERT"),
    "reasonir-8b": lambda tp: ReasonIRProcessor(),
    "reasonir-8b-vllm": lambda tp: VLLMProcessor.from_huggingface(
        "reasonir/ReasonIR-8B",
        hf_overrides={
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        pooler_config={"pooling_type": "MEAN", "normalize": True},
    ),
    "splade-code-8b": lambda tp: SpladeProcessor(),
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
    "rader-reranker-7b": lambda tp: RaDeRRerankerVLLMProcessor(),
    "diver-grouprank-32b": lambda tp: GroupRankProcessor(
        tensor_parallel_size=max(1, tp)
    ),
    "inf-retriever-v1-pro": lambda tp: _build_inf_retriever_processor(),
    "inf-x-retriever": lambda tp: INFXRetrieverProcessor(
        rewrite_log_path="results/evaluation/.rewrites/inf-x-retriever.json"
    ),
    "retro-star-32b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-32b-0928", tensor_parallel_size=max(1, tp)
    ),
    "retro-star-8b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-8b-0928", tensor_parallel_size=max(1, tp)
    ),
    "reason-rewriter-reason-embed-8b": lambda tp: ReasonRewriterProcessor(
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        )
    ),
    "reason-rewriter-reason-embed-llama-3.1-8b": (
        lambda tp: ReasonRewriterProcessor(
            retriever_name="hanhainebula/reason-embed-llama-3.1-8b-0928",
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            ),
            retriever_init_kwargs={"max_model_len": 40960},
        )
    ),
    "reason-rewriter-reason-embed-8b-instructed": (
        lambda tp: ReasonRewriterProcessor(
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            )
        )
    ),
    "retro-star-8b-rewritten": lambda tp: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-8b-0928",
        tensor_parallel_size=max(1, tp),
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
    ),
    "retro-star-32b-rewritten": lambda tp: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-32b-0928",
        tensor_parallel_size=max(1, tp),
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
        gpu_memory_utilization=0.85,
        require_cached_rewrites=True,
    ),
}

CUSTOM_MODELS_USE_TP = {
    "diver-grouprank-32b",
    "retro-star-32b",
    "retro-star-8b",
    "retro-star-8b-rewritten",
    "retro-star-32b-rewritten",
}
_RADER_BIENCODER_SCORES_KWARGS = {
    "chunk_to_context": True,
    "context_length": 2048,
    "query_prompt": "query: ",
    "document_prompt": "document: ",
    "query_suffix": RADER_EXPECTED_EOS,
    "document_suffix": RADER_EXPECTED_EOS,
}

CUSTOM_MODEL_SCORES_KWARGS = {
    "rader-3b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-7b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-14b": dict(_RADER_BIENCODER_SCORES_KWARGS),
}

GENERIC_MODELS = {
    "qwen3-embedding-8b": {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True},
    "reason-embed-qwen3-8b": {
        "model": "hanhainebula/reason-embed-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "reason-embed-basic-qwen3-8b": {
        "model": "hanhainebula/reason-embed-basic-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "reason-embed-qwen3-4b": {
        "model": "hanhainebula/reason-embed-qwen3-4b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "reason-embed-llama-3.1-8b": {
        "model": "hanhainebula/reason-embed-llama-3.1-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "max_model_len": 40960,
        },
    },
    "rtriever-4b": {
        "model": "yale-nlp/RTriever-4B",
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
    "diver-retriever-0.6b": {
        "model": "AQ-MedAI/Diver-Retriever-0.6B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "dtype": "bfloat16",
        },
    },
}
GENERIC_MODELS_USE_TP = {"qwen3-embedding-8b"}

PIPELINE_MODEL_KEYS = list(CUSTOM_MODEL_BUILDERS) + list(GENERIC_MODELS)

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
    "llama-embed-nemotron-8b": {
        "model": "nvidia/llama-embed-nemotron-8b",
        "use_vllm": True,
    },
    "kalm-embedding-gemma3-12b-2511": {
        "model": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "use_vllm": True,
    },
    "embeddinggemma-300m": {
        "model": "google/embeddinggemma-300m",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "task: search result | query: ",
            "document_prompt": "title: none | text: ",
        },
    },
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
    # Deliberately prompt-free, against the model card:
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

API_MODEL_SCORES_KWARGS = {
    "gemini-embedding-001": {
        "query_encode_kwargs": {"task_type": "RETRIEVAL_QUERY"},
        "document_encode_kwargs": {"task_type": "RETRIEVAL_DOCUMENT"},
    },
}

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
    "bm25": (
        "dropped from the instruction experiment. BM25 has no "
        "instruction mechanism - the prompt can only be prepended to the "
        "query as more query terms - and unlike its two lexical siblings it "
        "has no pathology that makes the result uninterpretable, so it ran "
        "and scored 0.4165 -> 0.3568/0.3630/0.3881 on statement-full. "
        "Reporting one lexical baseline with arms and two without is a "
        "harness accident, not a distinction, so all three are now excluded "
        "together and the runs were deleted"
    ),
    "bm25-no-tok": ("same reason as bm25"),
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

ALL_MODEL_KEYS = (
    PIPELINE_MODEL_KEYS
    + list(TABLE_MODELS)
    + list(API_MODELS)
    + list(LEXICAL_MODEL_BUILDERS)
)



def model_spec(model_key: str) -> dict | None:
    if model_key in GENERIC_MODELS:
        return GENERIC_MODELS[model_key]
    if model_key in TABLE_MODELS:
        return TABLE_MODELS[model_key]
    return None


def model_scores_kwargs(model_key: str) -> dict:
    if model_key in CUSTOM_MODEL_SCORES_KWARGS:
        return dict(CUSTOM_MODEL_SCORES_KWARGS[model_key])
    if model_key in API_MODEL_SCORES_KWARGS:
        return dict(API_MODEL_SCORES_KWARGS[model_key])
    spec = model_spec(model_key)
    return dict(spec.get("scores_kwargs", {})) if spec is not None else {}


def prompt_scores_kwargs(
    model_key: str, instruction: str | None = None
) -> tuple[dict, bool]:
    kwargs = model_scores_kwargs(model_key)
    wrap_instruction = True

    if model_key == "reasonir-8b" and instruction is not None:
        wrap_instruction = False
        kwargs["query_encode_kwargs"] = {
            "instruction": REASONIR_QUERY_INSTRUCTION_SLOT.format(
                instruction=instruction
            )
        }
        kwargs["document_encode_kwargs"] = {"instruction": ""}

    if (
        model_key == "reason-rewriter-reason-embed-8b-instructed"
        and instruction is not None
    ):
        kwargs["embed_instruction"] = instruction

    return kwargs, wrap_instruction


def processor_slot(model_key: str, instruction_key: str) -> str:
    if model_key in QWEN3_RERANKER_REPOS:
        return f"qwen3__{instruction_key}"
    return "default"


def uses_tensor_parallel(model_key: str) -> bool:
    return model_key in GENERIC_MODELS_USE_TP or model_key in CUSTOM_MODELS_USE_TP


def is_control_model(model_key: str) -> bool:
    return model_key in INSTRUCTION_CONTROL_MODELS


def build_spec_processor(spec: dict, model_key: str, tensor_parallel_size: int):
    spec = dict(spec)
    model_name = spec.pop("model")
    init_kwargs = dict(spec.pop("init_kwargs", {}))
    use_vllm = spec.pop("use_vllm", False)
    disable_st_default_prompt = spec.pop("disable_st_default_prompt", False)
    if not use_vllm:
        processor = STProcessor.from_huggingface(model_name, **init_kwargs)
        if disable_st_default_prompt:
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


def build_processor(
    model_key: str,
    instruction_key: str,
    tensor_parallel_size: int = 1,
    save_dir: str | Path = "results/evaluation",
):
    instruction = INSTRUCTIONS[instruction_key]

    if model_key in QWEN3_RERANKER_REPOS:
        task_instruction = instruction or VENDOR_QWEN3_RERANKER_INSTRUCTION
        return Qwen3RerankerVLLMProcessor(
            QWEN3_RERANKER_REPOS[model_key], task_instruction=task_instruction
        )
    if model_key == "diver-grouprank-32b":
        return GroupRankProcessor(
            tensor_parallel_size=max(1, tensor_parallel_size),
            scaffold_reserve_tokens=EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE,
        )
    if model_key in COLBERT_REPOS:
        return ColBERTProcessor(
            COLBERT_REPOS[model_key], query_length=EXPERIMENT_COLBERT_QUERY_LENGTH
        )
    if model_key == "inf-x-retriever":
        return INFXRetrieverProcessor(
            rewrite_log_path=str(
                Path(save_dir) / ".rewrites" / "inf-x-retriever.json"
            )
        )
    if model_key in CUSTOM_MODEL_BUILDERS:
        return CUSTOM_MODEL_BUILDERS[model_key](tensor_parallel_size)
    spec = model_spec(model_key)
    if spec is not None:
        return build_spec_processor(spec, model_key, tensor_parallel_size)
    if model_key in API_MODELS:
        kind, model_name = API_MODELS[model_key]
        if kind == "google":
            return GoogleProcessor(model_name)
        if kind == "openrouter":
            return OpenRouterEmbeddingProcessor(model_name)
        return OpenAIProcessor(model_name)
    if model_key in LEXICAL_MODEL_BUILDERS:
        return LEXICAL_MODEL_BUILDERS[model_key]()
    raise KeyError(f"Unknown model key: {model_key}")
