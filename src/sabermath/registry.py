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

# The p0 baseline for the Qwen3-Reranker family, so their p0 measures "no task
# instruction" like every other row. The repo's math-specific instruction is
# prompt key "pm".
VENDOR_QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Models with no vendor-documented free-text instruction mechanism: their
# p1-p3 rows are CONTROLS, not instruction-following measurements, and are
# reported separately. Distinct from INSTRUCTION_EXCLUDED, which hard-errors.
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

# ReasonIR prepends this to the query, then masks exactly these tokens out of
# the pool. With instruction="" it prepends nothing, which is why the document
# side and the whole p0 arm are bare text. See docs/protocol.md.
REASONIR_QUERY_INSTRUCTION_SLOT = "<|user|>\n{instruction}\n<|embed|>\n"

# The EOS the RaDeR bi-encoder envelope below appends; its last-token pooler
# reads that position, so a mismatched token silently changes every vector.
RADER_EXPECTED_EOS = "<|im_end|>"

# The underscore in the 3B repo name and the hyphen in the 7B/14B are the
# upstream names, not a typo.
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
    # Dtype ablation of rank1-32b, not a separate model: the production spec
    # pins fp16 per the model card, while the checkpoint is bf16. Dtype is the
    # only difference between this key and rank1-32b.
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
    # Official remote-code path, not vLLM: vLLM's stock MEAN pooler cannot
    # express ReasonIR's instruction masking. See docs/backend-provenance.md.
    "reasonir-8b": lambda tp: ReasonIRProcessor(),
    # p0-only vLLM variant of reasonir-8b, kept as a separate key so the entry
    # above stays available for the instructed arms. The hf_overrides redirect
    # is needed because vLLM does not register ReasonIR's declared
    # architecture. INSTRUCTION_EXCLUDED hard-errors this key on p1-p3.
    "reasonir-8b-vllm": lambda tp: VLLMProcessor.from_huggingface(
        "reasonir/ReasonIR-8B",
        hf_overrides={
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        pooler_config={"pooling_type": "MEAN", "normalize": True},
    ),
    "splade-code-8b": lambda tp: SpladeProcessor(),
    # The repo name really has no dot; "naver/splade-code-0.6B" does not exist.
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
    # The one bi-encoder deliberately left on SentenceTransformers: its
    # bidirectional remote-code attention has no validated vLLM path.
    "inf-retriever-v1-pro": lambda tp: _build_inf_retriever_processor(),
    # The full infly system: an aligner rewrites the query, then the REWRITE is
    # embedded through the same retriever backend as the standalone entry
    # above, so the two rows isolate the rewrite. See INFXRetrieverProcessor.
    "inf-x-retriever": lambda tp: INFXRetrieverProcessor(
        rewrite_log_path="results/evaluation/.rewrites/inf-x-retriever.json"
    ),
    # A generative pointwise reranker: its score is an integer the model writes
    # into a <score> tag, not a logprob. See RetroStarProcessor. The 32B is the
    # most expensive entry here - a full generative pass per query-document
    # PAIR - so give it every GPU on the node and expect to resume it.
    "retro-star-32b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-32b-0928", tensor_parallel_size=max(1, tp)
    ),
    "retro-star-8b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-8b-0928", tensor_parallel_size=max(1, tp)
    ),
    # The BGE-Reasoner "rewrite, then embed" system: five sampled rewrites per
    # query, embeddings mean-pooled. Stands to reason-embed-qwen3-8b as
    # inf-x-retriever stands to inf-retriever-v1-pro, so the pair of rows
    # isolates the rewrite. See ReasonRewriterProcessor.
    "reason-rewriter-reason-embed-8b": lambda tp: ReasonRewriterProcessor(
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        )
    ),
    # The same system with the LLaMA-3.1 Reason-Embed as the retriever half.
    # Sharing the Qwen3 row's rewrite log is safe, not just convenient: only
    # the embedder differs, the log holds no embeddings, and _load_rewrite_log
    # revalidates the rewriter, task string and recipe fingerprint before
    # trusting an entry. max_model_len mirrors the standalone spec, without
    # which vLLM would take llama-3.1's 131072 default and the two rows would
    # no longer be preprocessing-identical.
    "reason-rewriter-reason-embed-llama-3.1-8b": (
        lambda tp: ReasonRewriterProcessor(
            retriever_name="hanhainebula/reason-embed-llama-3.1-8b-0928",
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            ),
            retriever_init_kwargs={"max_model_len": 40960},
        )
    ),
    # The same system with the instruction reaching BOTH halves. On the row
    # above a prompt only reaches the rewriter, because the encoder embeds the
    # rewrites; this key reapplies the wrap to each rewrite before embedding.
    # The three arms are therefore: encoder-only (standalone
    # reason-embed-qwen3-8b), rewriter-only (the row above), and both (here).
    # At p0 this row and the one above are the same measurement by
    # construction, since there is no wrap to reapply.
    "reason-rewriter-reason-embed-8b-instructed": (
        lambda tp: ReasonRewriterProcessor(
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            )
        )
    ),
    # Retro* scoring each pair against the REWRITTEN query. This is NOT the
    # vendor's cascade, which reranks the original query and uses the rewrite
    # only for retrieval - on this benchmark's fixed candidate sets that would
    # reproduce retro-star-8b exactly. Read RetroStarRewrittenProcessor before
    # reporting this number.
    "retro-star-8b-rewritten": lambda tp: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-8b-0928",
        tensor_parallel_size=max(1, tp),
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
    ),
    # Same rewrite-as-the-reranker's-query construction at 32B. Read the
    # warning on retro-star-8b-rewritten first: this is NOT the vendor's
    # cascade, and its number must not be reported as one.
    #
    # Two differences from the 8B entry, both because of the size. The
    # reranker gets 0.85 of the GPU instead of 0.55 - a 32B in bf16 is ~64GB
    # of weights, and at 0.55 there is almost no KV cache left, which with
    # ~11k-token prompts (a ~4k-token rewrite plus the document) would leave
    # one or two candidates in flight. That budget only works because every
    # rewrite is already cached and the 7B generator never loads, so
    # require_cached_rewrites makes the alternative a clear error rather than
    # an OOM. Run this SHARDED like plain retro-star-32b (--query-shards);
    # longer prompts make it slower per query than that row.
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

# Custom models where --tensor-parallel-size is actually honored (everything
# else is either pinned to 1 for correctness - rank1-32b - or a single-GPU
# HF/PyLate call that can't shard at all).
CUSTOM_MODELS_USE_TP = {
    "diver-grouprank-32b",
    "retro-star-32b",
    "retro-star-8b",
    "retro-star-8b-rewritten",
    "retro-star-32b-rewritten",
}

# Per-model scores_kwargs, forwarded to processor.get_scores(). The
# *_prompt/*_suffix entries are the vendor-documented input envelope
# (docs/protocol.md), applied per side before the vector cache, so an affixed
# text is simply a new cache key. RaDeR's own retrievers.py builds
# "query: {q}<|im_end|>" / "document: {d}<|im_end|>" with last-token pooling.
# context_length must stay fixed across backends and sizes or the arms are no
# longer preprocessing-identical.
_RADER_BIENCODER_SCORES_KWARGS = {
    "chunk_to_context": True,
    "context_length": 2048,
    "query_prompt": "query: ",
    "document_prompt": "document: ",
    "query_suffix": RADER_EXPECTED_EOS,
    "document_suffix": RADER_EXPECTED_EOS,
}
# ReasonIR gets no text envelope: its own encode() adds nothing when the
# instruction is empty. Its instructed arms use per-side encode kwargs.
CUSTOM_MODEL_SCORES_KWARGS = {
    "rader-3b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-7b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-14b": dict(_RADER_BIENCODER_SCORES_KWARGS),
}

# Models needing no custom processor - just the right (model name, use_vllm)
# pair through evaluate()'s generic HuggingFace path. init_kwargs are passed
# straight to vllm.LLM(). The explicit LAST+normalize pooler overrides are
# deterministic rather than trusting the architecture defaults.
GENERIC_MODELS = {
    "qwen3-embedding-8b": {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True},
    "reason-embed-qwen3-8b": {
        "model": "hanhainebula/reason-embed-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    # A backbone/size ablation around reason-embed-qwen3-8b, so all four arms
    # are configured identically on purpose: an ablation is only readable if
    # preprocessing matches across it. No dtype pin, because every repo in the
    # family declares float32 and vLLM downcasts pooling models to fp16
    # anyway; no query envelope, because the prompt these repos ship is the
    # same shape the instruction experiment supplies, so p0 stays prompt-free
    # and the prompt is studied there instead.
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
    # max_model_len is the one deliberate difference from its siblings, and it
    # is result-neutral: the Llama-3.1 config advertises 131072 positions where
    # the Qwen3 arms advertise 40960, and vLLM would size its batches and
    # profiling run for a 131k-token batch this benchmark never produces.
    # Nothing here is truncated at 40960 - the longest document is ~6k tokens.
    "reason-embed-llama-3.1-8b": {
        "model": "hanhainebula/reason-embed-llama-3.1-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "max_model_len": 40960,
        },
    },
    # Same Qwen3 architecture family as the entries above, so the same pooler
    # override; run prompt-free in the production arm like the rest of that
    # family, with the instruction dimension studied through --prompts.
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
    # Size-ablation sibling of diver-retriever-4b, so identical init_kwargs.
    # dtype must stay pinned: this repo's config declares none, so vLLM's
    # "auto" runs it fp16 while the bf16-declaring 4B sibling runs bf16,
    # splitting precision within one ablation family.
    "diver-retriever-0.6b": {
        "model": "AQ-MedAI/Diver-Retriever-0.6B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "dtype": "bfloat16",
        },
    },
}
GENERIC_MODELS_USE_TP = {"qwen3-embedding-8b"}  # vLLM-backed -> tp actually used

# The keys the per-family conda environment rules apply to - see
# docs/experiment-evaluation.md. The API and lexical keys are exempt.
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
    # Must stay on vLLM. This model is bidirectional only through an override
    # of LlamaModel._update_causal_mask(), which transformers 5.x deleted - so
    # under SentenceTransformers with an unpinned transformers it silently runs
    # CAUSAL and reports nothing. Setting use_vllm: False here without ALSO
    # pinning transformers <5 and forcing attn_implementation reintroduces
    # that. See docs/backend-provenance.md.
    "llama-embed-nemotron-8b": {
        "model": "nvidia/llama-embed-nemotron-8b",
        "use_vllm": True,
    },
    "kalm-embedding-gemma3-12b-2511": {
        "model": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "use_vllm": True,
    },
    # EmbeddingGemma is a prompt-template-trained model: its documented
    # grammar is a fixed closed set, not free text.
    "embeddinggemma-300m": {
        "model": "google/embeddinggemma-300m",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "task: search result | query: ",
            "document_prompt": "title: none | text: ",
        },
    },
    # jina-v5 ships prompts {"query": "Query: ", "document": "Document: "}
    # with default_prompt_name="document" - so the prompt-less ST path used
    # to prepend "Document: " to QUERIES as well as documents.
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
    # Must stay on vLLM. An earlier hand-built [Transformer, Pooling(lasttoken),
    # Normalize] imitation of this vendor's stack agreed with vLLM only at
    # cosine 0.70, where a generic SentenceTransformer load agrees at 0.9998 -
    # the imitation was the outlier, not the engine. See
    # docs/backend-provenance.md.
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
    # Deliberately prompt-free, against the model card: E5 documents
    # "query: "/"passage: " as required, but on this benchmark they cost
    # 0.028-0.080 nDCG. The deviation is disclosed - see docs/protocol.md.
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

# Gemini's retrieval mechanism is the task_type enum, not a text prefix.
# gemini-embedding-001 only: on -2, every valid task_type returns a
# byte-identical vector, so sending one would buy nothing and would split one
# embed call into two. text-embedding-3-* have no mechanism at all. Probe
# output: results/diagnostics/protocol/gemini.json.
API_MODEL_SCORES_KWARGS = {
    "gemini-embedding-001": {
        "query_encode_kwargs": {"task_type": "RETRIEVAL_QUERY"},
        "document_encode_kwargs": {"task_type": "RETRIEVAL_DOCUMENT"},
    },
}

# The "-no-tok" keys are each method in its own default tokenization; the
# plain keys run Approach Zero's pya0 tokenizer over every LaTeX block (see
# tokenization_helper.math_word_tokens). Both are reported, because the math
# tokenizer is not uniformly better: it gains BM25 and Jaccard, costs TF-IDF.
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

# Every evaluable key, and what `--models` defaults to.
ALL_MODEL_KEYS = (
    PIPELINE_MODEL_KEYS
    + list(TABLE_MODELS)
    + list(API_MODELS)
    + list(LEXICAL_MODEL_BUILDERS)
)




# ---------------------------------------------------------------------------
# Lookups. Pure functions over the tables above - no argv, no filesystem - so
# the report generators can ask the same questions the runner does.
# ---------------------------------------------------------------------------


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
        # The official mechanism, exactly: prepend
        # "<|user|>\n{task}\n<|embed|>\n" to the query and then exclude
        # precisely those tokens from the mean pool. ReasonIRProcessor.encode
        # takes that whole prefix as its `instruction` argument and does the
        # masking itself, so it goes through the per-side encode kwargs
        # rather than as a text affix - and the generic Instruct:/Query: wrap
        # must not be applied on top of it.
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
        # wrap_instruction stays True: the wrap around the query text is what
        # instructs the REWRITER, and it is also the rewrite cache key, so
        # both composed arms hit the same cached rewrites. This kwarg is the
        # ENCODER half - get_scores reapplies the identical wrap to each
        # rewrite before embedding. Deliberately NOT an AFFIX_KEYS member:
        # those are applied by EmbeddingProcessor, and the envelope assertion
        # in the runner would rightly reject one here, since this processor is
        # a composed ModelProcessor.
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
            # This model's prompts are applied explicitly per side via
            # scores_kwargs; leaving default_prompt_name set would make
            # sentence-transformers prepend its own default on top.
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
        # This family has a genuine <Instruct> slot, so p0 must fill it with
        # the VENDOR default to be a real "no task instruction" baseline; the
        # repo's production math instruction lives on as prompt key "pm".
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
        # ColBERT pads queries to query_length with mask tokens (query
        # augmentation), so an arm-dependent query_length would change the
        # query representation even for identical text.
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
