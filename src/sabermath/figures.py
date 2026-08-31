# The model roster, display names, markers and colours shared by every
# figure, so the math-vs-word and latency plots name and colour a model
# identically. Previously scraped out of the plotting script's source with
# ast; both figures import it directly now.

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
    # Every id must match load_models' own strings EXACTLY -
    # model_to_file_stem() only does .replace("/", "_").
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
    # The reason-family standalone/rewritten pairs. Each rewritten row is the
    # same scorer reading a rewrite of the query, so the PAIR is what makes the
    # rewrite's effect readable. LIST ORDER IS LEGEND ORDER, and the legend
    # fills columns top-to-bottom, so keep this block contiguous and short
    # enough for one column or the family splits across a column break.
    "hanhainebula/reason-embed-qwen3-8b-0928",
    "hanhainebula/reason-embed-llama-3.1-8b-0928",
    "reason-rewriter-reason-embed-8b",
    "retro-star-32b",
    "retro-star-32b-rewritten",
]

MATH_TOKEN_RATIO_MODEL_ID = "target_math_token_ratio"

DEFAULT_MODEL_DISPLAY_NAMES = {
    # Labels track the paper's main results table, capitalization quirks
    # included, so the figure and the table name each system identically.
    # Models with no row there are marked "not in the table" and follow the
    # same conventions.
    "approach0": "Approach Zero",
    "Octen/Octen-Embedding-8B": "Octen-8B",
    "Octen/Octen-Embedding-4B": "Octen-4B",  # not in the table
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
    "microsoft/codebert-base": "CodeBERT",  # not in the table
    "hanhainebula/reason-embed-qwen3-8b-0928": "ReasonEmbed-Qwen3-8B",
    "AQ-MedAI/Diver-Retriever-4B": "Diver-4B",
    "AQ-MedAI/Diver-Retriever-0.6B": "Diver-0.6B",  # not in the table
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
    "jhu-clsp/rank1-7b": "Rank1-7B",  # not in the table
    "jhu-clsp/rank1-0.5b": "Rank1-0.5B",  # not in the table
    "inf-x-retriever": "INF-X-Retriever",
    "qwen3-reranker-8b": "Qwen3-Reranker-8B",  # not in the table
    "qwen3-reranker-4b": "Qwen3-Reranker-4B",
    "qwen3-reranker-0.6b": "Qwen3-Reranker-0.6B",
    "splade-code-8b": "SPLADE-Code-8B",
    "splade-code-0.6b": "SPLADE-Code-0.6B",  # not in the table
    "rader-reranker-7b": "RaDeR-Reranker-7B",  # not in the table
    "diver-grouprank-32b": "Diver-GroupRank-32B",
    # The table names Reason-ModernColBERT "Reason-ColBERT"; GTE's sibling
    # has no row and takes the same shortening.
    "lightonai/GTE-ModernColBERT-v1": "GTE-ColBERT",  # not in the table
    "lightonai/Reason-ModernColBERT": "Reason-ColBERT",
    # The key is the checkpoint's own name, which is also the filename on
    # disk; the label follows the paper, which calls it ReasonReranker.
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "ReasonEmbed-Llama-3.1-8B",
    # A rewritten row SUFFIXES its scorer's label with "-Rewrite" rather than
    # prefixing the rewriter's name, so the pair sorts together and differs by
    # one word - which is what makes the rewrite read as one marker moving.
    # results/rescaling/results.json names them the same way.
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
    # Lexical/string baselines
    "tf-idf": "*",
    "jaccard": "*",
    "bm25": "*",
    # RaDeR family (Raderspace bi-encoders + its LoRA reranker)
    "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes": "d",
    "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes": "d",
    "rader-reranker-7b": "d",
    # Diver family (AQ-MedAI retrievers + the groupwise reranker)
    "AQ-MedAI/Diver-Retriever-4B": "<",
    "AQ-MedAI/Diver-Retriever-0.6B": "<",
    "diver-grouprank-32b": "<",
    # Qwen-architecture derivatives - share Qwen's diamond marker
    "hanhainebula/reason-embed-qwen3-8b-0928": "D",
    "qwen3-reranker-8b": "D",
    "qwen3-reranker-4b": "D",
    "qwen3-reranker-0.6b": "D",
    # SPLADE (naver)
    "splade-code-8b": "P",
    "splade-code-0.6b": "P",
    # ColBERT (lightonai)
    "lightonai/GTE-ModernColBERT-v1": "h",
    "lightonai/Reason-ModernColBERT": "h",
    # matplotlib's filled-shape alphabet (o v ^ < > 8 s p * h H D d P X) is
    # exhausted above, so these reuse a shape and rely on colour alone. Do NOT
    # reach for '1'/'2'/'3'/'4'/'+'/'x'/'|'/'_': they are line-only markers,
    # and every scatter here sets edgecolors="none", which draws NOTHING.
    "jhu-clsp/rank1-32b": ">",
    "jhu-clsp/rank1-7b": ">",  # shares rank1-32b's triangle_right
    "jhu-clsp/rank1-0.5b": ">",  # shares rank1-32b's triangle_right
    "inf-x-retriever": "8",  # shares inf-retriever-v1-pro's octagon
    "infly/inf-retriever-v1-pro": "8",  # shares BAAI/bge-m3's octagon
    "nvidia/llama-embed-nemotron-8b": "v",  # shares RoBERTa's triangle_down
    "intfloat/multilingual-e5-large": "^",  # shares the Google family's triangle_up
    "reasonir/ReasonIR-8B": "o",  # shares approach0's circle
    "jinaai/jina-embeddings-v5-text-small": "p",
    # OpenAI (via OpenRouter) - shares Octen's square
    "text-embedding-3-large": "s",
    "text-embedding-3-small": "s",
    # The Reason-Embed family shares one diamond, so a standalone row and its
    # rewritten counterpart differ only in shade.
    "hanhainebula/reason-embed-llama-3.1-8b-0928": "D",
    "reason-rewriter-reason-embed-8b": "D",
    # Shares KaLM's hexagon2 - the olive/lime below is far from KaLM's cyan.
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
