import argparse
from collections import defaultdict
import json
import random
import os

from datasets import load_dataset
import numpy as np
import torch

import sabermath

QUERY_DATASET_PATH = "sabermath/SaberMath-queries"
DOMAINS = [
    "Algebra",
    "Geometry",
    "Number Theory",
    "Combinatorics",
    "Calculus and Analysis",
]
DOMAIN_SAMPLE_COUNT = 300
X = 10000
SEED = 42411

CACHE_DIR = "../../.vector.cache"
RESULT_DIR = "./confresult"

cached_models = [
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-0.6b",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "nvidia/llama-embed-nemotron-8b",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "FacebookAI/roberta-base",
    "google-bert/bert-base-uncased",
    "google/embeddinggemma-300m",
    "intfloat/multilingual-e5-large",
    "microsoft/harrier-oss-v1-27b",
    "jinaai/jina-embeddings-v5-text-nano",
    "jinaai/jina-embeddings-v5-text-small",
    "Octen/Octen-Embedding-4B",
    "Octen/Octen-Embedding-8B",
    "gemini-embedding-001",
    "gemini-embedding-2",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

special_models = {
    "tf-idf": sabermath.processors.TfidfProcessor,
    "jaccard": sabermath.processors.JaccardProcessor,
    "bm25": sabermath.processors.BM25Processor,
    "approach0": sabermath.processors.Approach0Processor,
}

QUERY_DATASET = load_dataset(QUERY_DATASET_PATH, split="train")


def get_domain_idxs(domain: str) -> list[int]:
    return [i for i, row in enumerate(QUERY_DATASET) if domain in row["domains"]]


DOMAIN_IDXS = {domain: get_domain_idxs(domain) for domain in DOMAINS}

ALL_IDXS = list(range(len(QUERY_DATASET)))


def get_task_confidence_sample(ndcgs, task, rng):
    task_ndcgs = np.asarray(ndcgs[task], dtype=object)

    def valid_idxs(candidate_idxs):
        candidate_idxs = np.asarray(candidate_idxs)
        return np.asarray(
            [idx for idx in candidate_idxs if task_ndcgs[idx] is not None]
        )

    def sample_valid(candidate_idxs, sample_count):
        idxs = valid_idxs(candidate_idxs)

        if len(idxs) == 0:
            raise ValueError(f"No valid NDCG values available for task={task}")

        return np.asarray([rng.choice(idxs) for _ in range(sample_count)])

    domain_idxs_sampled = {
        domain: sample_valid(DOMAIN_IDXS[domain], DOMAIN_SAMPLE_COUNT)
        for domain in DOMAINS
    }

    full_idx_sampled = sample_valid(ALL_IDXS, len(task_ndcgs))

    results = {
        domain: float(np.mean(task_ndcgs[idxs].astype(float)))
        for domain, idxs in domain_idxs_sampled.items()
    }

    results["total"] = float(np.mean(task_ndcgs[full_idx_sampled].astype(float)))

    return results


def confidence_interval_95(values):
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    return [float(lower), float(upper)]


def get_task_confidence(ndcgs, task, X: int):
    rng = random.Random(SEED)

    samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(X):
        d = get_task_confidence_sample(ndcgs, task, rng)
        for key, value in d.items():
            samples[key].append(value)

    results = {key: confidence_interval_95(values) for key, values in samples.items()}

    branch_dict = [
        {
            "branch": domain,
            "confidence_interval": results[domain],
            "mean": np.mean(samples[domain]),
        }
        for domain in DOMAINS
    ]

    task_dict = {
        "task": task,
        "confidence_interval": results["total"],
        "mean": np.mean(samples["total"]),
        "branches": branch_dict,
    }

    return task_dict


def format_name(model: str):
    return model.replace("/", "_")


def get_confidence_cached_model(model, tasks):
    name = format_name(model) + ".npz"
    cache_path = os.path.join(CACHE_DIR, name)

    init_kwargs = {
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": torch.bfloat16},
    }

    if use_vllm:
        cache_path = None

    _, ndcgs = sabermath.evaluate(
        model,
        tasks=tasks,
        use_vllm=use_vllm,
        cache_path=cache_path,
        return_ndcgs=True,
        verbose=False,
        init_kwargs=init_kwargs,
        no_init=True,
    )

    total_dict = {
        "k": 10,
        "tasks": [get_task_confidence(ndcgs, task, X=10000) for task in tasks],
    }

    return total_dict


def get_confidence_special_model(model_name, tasks):
    cls_ = special_models.get(model_name)

    if cls_ is None:
        raise ValueError("Invalid model name.")

    model = cls_()

    _, ndcgs = sabermath.evaluate(
        model,
        return_ndcgs=True,
        verbose=False,
        no_init=True,
    )

    total_dict = {
        "k": 10,
        "tasks": [get_task_confidence(ndcgs, task, X=10000) for task in tasks],
    }

    return total_dict


def main() -> None:
    tasks = ["statement-full"]

    os.makedirs(RESULT_DIR, exist_ok=True)

    cml = len(cached_models)
    sml = len(special_models)
    tml = cml + sml

    if cml:
        print(f" ========== RUNNING CACHED MODELS ({cml}) ==========")

    for i, cmodel in enumerate(cached_models, start=1):
        print(f"Running {cmodel}... {i}/{tml}")

        result = get_confidence_cached_model(cmodel, tasks)

        result_path = os.path.join(
            RESULT_DIR,
            f"{format_name(cmodel)}.json",
        )

        with open(result_path, "w") as file:
            json.dump(result, file, indent=2)

    if cml:
        print("\n\n")

    print(f" ========== RUNNING SPECIAL MODELS ({sml}) ==========")

    for i, name in enumerate(special_models.keys(), start=cml):
        print(f"Running {name}... {i}/{tml}")

        result = get_confidence_special_model(name, tasks)

        result_path = os.path.join(
            RESULT_DIR,
            f"{format_name(name)}.json",
        )

        with open(result_path, "w") as file:
            json.dump(result, file, indent=2)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
