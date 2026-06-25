from vllm import LLM
import torch
import os
from sentence_transformers import SentenceTransformer

# Set CUDA devices strictly if needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_model(MODEL_ID: str) -> dict:

    d = {}

    if any(
        x in MODEL_ID
        for x in ["embeddinggemma", "bert", "roberta", "codebert", "jinaai"]
    ):
        d["type"] = "sentence-transformers"
    elif (
        MODEL_ID == "google/gemini-embedding-001"
        or MODEL_ID == "google/gemini-embedding-2"
    ):
        d["type"] = "google"
    else:
        d["type"] = "vllm"

    print(f"Model type is {d['type']}")

    print(f"Loading {MODEL_ID} using backend: {d['type']}...")

    if d["type"] == "sentence-transformers":

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float32} if device == "cuda" else {}

        model = SentenceTransformer(
            MODEL_ID, device=device, model_kwargs=model_kwargs, trust_remote_code=True
        )

    elif d["type"] == "vllm":

        vllm_args = {
            "model": MODEL_ID,
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "enforce_eager": True,
            "dtype": "bfloat16",
        }

        if "tencent" in MODEL_ID:
            vllm_args["revision"] = "CausalLM"

        model = LLM(**vllm_args)

    elif d["type"] == "google":
        model = None

    d["model"] = model

    return d


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
