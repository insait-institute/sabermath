ALL_MODEL_IDS = [
    "approach0",
    "Octen/Octen-Embedding-8B",
    "Octen/Octen-Embedding-4B",
    "google/gemini-embedding-2",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "microsoft/harrier-oss-v1-27b",
    "google/gemini-embedding-001",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Qwen/Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-0.6b",
    "jinaai/jina-embeddings-v5-text-nano",
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
    "microsoft/harrier-oss-v1-270m",
    "tf-idf",
    "jaccard",
    "bm25",
    "google-bert/bert-base-uncased",
    "FacebookAI/roberta-base",
    "AQ-MedAI/Diver-Retriever-4B",
    "AQ-MedAI/Diver-Retriever-0.6B",
    "infly/inf-retriever-v1-pro",
    "reasonir/ReasonIR-8B",
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "nvidia/llama-embed-nemotron-8b",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v5-text-small",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "jhu-clsp/rank1-32b",
    "jhu-clsp/rank1-7b",
    "jhu-clsp/rank1-0.5b",
    "inf-x-retriever",
    "qwen3-reranker-8b",
    "qwen3-reranker-4b",
    "qwen3-reranker-0.6b",
    "splade-code-8b",
    "splade-code-0.6b",
    "rader-reranker-7b",
    "diver-grouprank-32b",
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/Reason-ModernColBERT",
    "hanhainebula/reason-embed-qwen3-8b-0928",
    "hanhainebula/reason-embed-llama-3.1-8b-0928",
    "reason-rewriter-reason-embed-8b",
    "retro-star-32b",
    "retro-star-32b-rewritten",
]

MATH_TOKEN_RATIO_MODEL_ID = "target_math_token_ratio"

DEFAULT_MODEL_DISPLAY_NAMES = {
    "approach0": "Approach Zero",
    "Octen/Octen-Embedding-8B": "Octen-8B",
    "Octen/Octen-Embedding-4B": "Octen-4B",  
    "google/gemini-embedding-2": "Gemini-2",
    "Qwen/Qwen3-Embedding-4B": "Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
    "microsoft/harrier-oss-v1-27b": "Harrier-27b",
    "google/gemini-embedding-001": "Gemini-001",
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "KaLM-12B-2511",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "microsoft/harrier-oss-v1-0.6b": "Harrier-0.6b",
    "jinaai/jina-embeddings-v5-text-nano": "Jina-v5-Nano",
    "google/embeddinggemma-300m": "Gemma-300m",
    "BAAI/bge-m3": "BGE-m3",
    "microsoft/harrier-oss-v1-270m": "Harrier-270m",
    "tf-idf": "TF-IDF",
    "jaccard": "Jaccard",
    "bm25": "BM25",
    "google-bert/bert-base-uncased": "BERT (Base)",
    "FacebookAI/roberta-base": "RoBERTa",
    "microsoft/codebert-base": "CodeBERT",  
    "hanhainebula/reason-embed-qwen3-8b-0928": "ReasonEmbed-Qwen3-8B",
    "AQ-MedAI/Diver-Retriever-4B": "Diver-4B",
    "AQ-MedAI/Diver-Retriever-0.6B": "Diver-0.6B",  
    "infly/inf-retriever-v1-pro": "INF-Retriever-v1-Pro",
    "reasonir/ReasonIR-8B": "ReasonIR-8B",
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "RaDeR-14B",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "RaDeR-7B",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "RaDeR-3B",
    "nvidia/llama-embed-nemotron-8b": "LLaMa-Nemotron-8B",
    "intfloat/multilingual-e5-large": "Multilingual-E5-Large",
    "jinaai/jina-embeddings-v5-text-small": "Jina-v5-Small",
    "text-embedding-3-large": "Text-Embedding-3-Large",
    "text-embedding-3-small": "Text-Embedding-3-Small",
    "jhu-clsp/rank1-32b": "Rank1-32B",
    "jhu-clsp/rank1-7b": "Rank1-7B",  
    "jhu-clsp/rank1-0.5b": "Rank1-0.5B", 
    "inf-x-retriever": "INF-X-Retriever",
    "qwen3-reranker-8b": "Qwen3-Reranker-8B", 
    "qwen3-reranker-4b": "Qwen3-Reranker-4B",
    "qwen3-reranker-0.6b": "Qwen3-Reranker-0.6B",
    "splade-code-8b": "SPLADE-Code-8B",
    "splade-code-0.6b": "SPLADE-Code-0.6B", 
    "rader-reranker-7b": "RaDeR-Reranker-7B", 
    "diver-grouprank-32b": "Diver-GroupRank-32B",
    "lightonai/GTE-ModernColBERT-v1": "GTE-ColBERT", 
    "lightonai/Reason-ModernColBERT": "Reason-ColBERT",
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "ReasonEmbed-Llama-3.1-8B",
    "reason-rewriter-reason-embed-8b": "ReasonEmbed-Qwen3-8B-Rewrite",
    "retro-star-32b": "ReasonReranker-Qwen3-32B",
    "retro-star-32b-rewritten": "ReasonReranker-Qwen3-32B-Rewrite",
    MATH_TOKEN_RATIO_MODEL_ID: "Math-token ratio",
}

DEFAULT_MODEL_MARKER_SYMBOLS = {
    "approach0": "o",
    "Octen/Octen-Embedding-8B": "s",
    "Octen/Octen-Embedding-4B": "s",
    "google/gemini-embedding-2": "^",
    "google/gemini-embedding-001": "^",
    "google/embeddinggemma-300m": "^",
    "google-bert/bert-base-uncased": "^",
    "Qwen/Qwen3-Embedding-8B": "D",
    "Qwen/Qwen3-Embedding-4B": "D",
    "Qwen/Qwen3-Embedding-0.6B": "D",
    "microsoft/harrier-oss-v1-27b": "X",
    "microsoft/harrier-oss-v1-0.6b": "X",
    "microsoft/harrier-oss-v1-270m": "X",
    "microsoft/codebert-base": "X",
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "H",
    "jinaai/jina-embeddings-v5-text-nano": "p",
    "BAAI/bge-m3": "8",
    "FacebookAI/roberta-base": "v",
    "tf-idf": "*",
    "jaccard": "*",
    "bm25": "*",
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "d",
    "rader-reranker-7b": "d",
    "AQ-MedAI/Diver-Retriever-4B": "<",
    "AQ-MedAI/Diver-Retriever-0.6B": "<",
    "diver-grouprank-32b": "<",
    "hanhainebula/reason-embed-qwen3-8b-0928": "D",
    "qwen3-reranker-8b": "D",
    "qwen3-reranker-4b": "D",
    "qwen3-reranker-0.6b": "D",
    "splade-code-8b": "P",
    "splade-code-0.6b": "P",
    "lightonai/GTE-ModernColBERT-v1": "h",
    "lightonai/Reason-ModernColBERT": "h",
    "jhu-clsp/rank1-32b": ">",
    "jhu-clsp/rank1-7b": ">", 
    "jhu-clsp/rank1-0.5b": ">",
    "inf-x-retriever": "8",
    "infly/inf-retriever-v1-pro": "8",
    "nvidia/llama-embed-nemotron-8b": "v",
    "intfloat/multilingual-e5-large": "^",  
    "reasonir/ReasonIR-8B": "o", 
    "jinaai/jina-embeddings-v5-text-small": "p",
    "text-embedding-3-large": "s",
    "text-embedding-3-small": "s",
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "D",
    "reason-rewriter-reason-embed-8b": "D",
    "retro-star-32b": "H",
    "retro-star-32b-rewritten": "H",
}

DEFAULT_MODEL_COLORS = {
    "approach0": "#4B4848",
    # Octen models: vivid blue/cyan family
    "Octen/Octen-Embedding-8B": "#0066ff",
    "Octen/Octen-Embedding-4B": "#00b4d8",
    # Google models: bright green/lime/teal family
    "google/gemini-embedding-2": "#00a651",
    "google/gemini-embedding-001": "#7bd000",
    "google/embeddinggemma-300m": "#00c2a8",
    "google-bert/bert-base-uncased": "#38ef7d",
    # Qwen models: saturated purple/magenta family
    "Qwen/Qwen3-Embedding-8B": "#6a00ff",
    "Qwen/Qwen3-Embedding-4B": "#b100ff",
    "Qwen/Qwen3-Embedding-0.6B": "#ff4fd8",
    # Microsoft models: red/orange/yellow family
    "microsoft/harrier-oss-v1-27b": "#ff1744",
    "microsoft/harrier-oss-v1-0.6b": "#ff6d00",
    "microsoft/harrier-oss-v1-270m": "#ffb300",
    "microsoft/codebert-base": "#c51162",
    "tencent/KaLM-Embedding-Gemma3-12B-2511": "#00acc1",
    "jinaai/jina-embeddings-v5-text-nano": "#ec407a",
    "BAAI/bge-m3": "#8d6e63",
    # Meta/Facebook
    "FacebookAI/roberta-base": "#1877f2",
    # Lexical/string baselines: distinguishable neutral colors
    "tf-idf": "#5f6368",
    "jaccard": "#9e9e9e",
    "bm25": "#757575",
    # RaDeR family: deep red/maroon, darkest = largest model
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "#b71c1c",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "#e53935",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "#ff8a80",
    "rader-reranker-7b": "#880e4f",
    # Diver family: blue
    "AQ-MedAI/Diver-Retriever-4B": "#1565c0",
    "AQ-MedAI/Diver-Retriever-0.6B": "#42a5f5",
    "diver-grouprank-32b": "#0d47a1",
    # Qwen-architecture derivatives: purple/magenta family (shares the hue
    # with Qwen3-Embedding, distinguished by shade)
    "hanhainebula/reason-embed-qwen3-8b-0928": "#d500f9",
    "qwen3-reranker-8b": "#4a0080",
    "qwen3-reranker-4b": "#7b1fa2",
    "qwen3-reranker-0.6b": "#ce93d8",
    # SPLADE: teal
    "splade-code-8b": "#00838f",
    "splade-code-0.6b": "#26c6da",
    # ColBERT: brown/amber
    "lightonai/GTE-ModernColBERT-v1": "#6d4c41",
    "lightonai/Reason-ModernColBERT": "#a1887f",
    # Standalone entities
    "jhu-clsp/rank1-32b": "#c62828",
    "jhu-clsp/rank1-7b": "#e53935",
    "jhu-clsp/rank1-0.5b": "#ef9a9a",
    "inf-x-retriever": "#26a69a",
    "infly/inf-retriever-v1-pro": "#00695c",
    "nvidia/llama-embed-nemotron-8b": "#558b2f",
    "intfloat/multilingual-e5-large": "#546e7a",
    "reasonir/ReasonIR-8B": "#f57f17",
    "jinaai/jina-embeddings-v5-text-small": "#f48fb1",
    # OpenAI (via OpenRouter): indigo
    "text-embedding-3-large": "#1a237e",
    "text-embedding-3-small": "#3949ab",
    # Standalone/rewritten pairs: same hue, DARK = standalone, LIGHT =
    # rewritten, so the rewrite reads as one marker moving.
    "reason-rewriter-reason-embed-8b": "#ea80fc",
    # Reason-Embed LLaMA: violet - same family, separable from the magenta.
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "#5e35b1",
    # Retro*: olive/lime - the reds and ambers are already crowded.
    "retro-star-32b": "#827717",
    "retro-star-32b-rewritten": "#c0ca33",
    # Reference statistic: dashed line color
    MATH_TOKEN_RATIO_MODEL_ID: "#263238",
}

# Figure model id -> the model key used by results/evaluation and
# results/timing (and by sabermath.tables.MODEL_INFO).
#
# The figures address models by their Hugging Face id because that is what the
# similarity dumps are named after; the result files address them by the
# registry's short key. sabermath.registry knows both, but importing it pulls
# in torch and vLLM, which a plotting script should not need - so the mapping
# is derived here instead, by a rule plus the ids that do not follow it.
# Consumers validate the result against the keys they actually have, so a
# wrong or stale entry raises rather than dropping a model from a figure.
_MODEL_KEY_EXCEPTIONS = {
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "rader-14b",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "rader-7b",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "rader-3b",
    "lightonai/GTE-ModernColBERT-v1": "gte-moderncolbert",
    "lightonai/Reason-ModernColBERT": "reason-moderncolbert",
    "hanhainebula/reason-embed-qwen3-8b-0928": "reason-embed-qwen3-8b",
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "reason-embed-llama-3.1-8b",
}


def model_key_for_id(model_id: str) -> str:
    """The results-file key for a figure model id."""
    if model_id in _MODEL_KEY_EXCEPTIONS:
        return _MODEL_KEY_EXCEPTIONS[model_id]
    return model_id.rsplit("/", 1)[-1].lower()


MODEL_KEY_BY_ID = {model_id: model_key_for_id(model_id) for model_id in ALL_MODEL_IDS}
MODEL_ID_BY_KEY = {key: model_id for model_id, key in MODEL_KEY_BY_ID.items()}
