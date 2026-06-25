from typing import Literal, Sequence
import numpy as np


def cosine_similarity(query_emb: np.ndarray, docs_embs: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_emb)
    d_norms = np.linalg.norm(docs_embs, axis=1, keepdims=True)

    if q_norm == 0:
        raise ValueError("Query embedding has zero norm")

    if np.any(d_norms == 0):
        raise ValueError("At least one document embedding has zero norm")

    query_emb = query_emb / q_norm
    docs_embs = docs_embs / d_norms
    return docs_embs @ query_emb


def dcg_at_k(
    relevances: Sequence[float],
    k: int = 10,
    variant: Literal["linear", "exponent"] = "linear",
) -> float:
    relevances = np.asarray(relevances[:k], dtype=float)

    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))

    if variant == "linear":
        gains = relevances
    elif variant == "exponent":
        gains = np.power(2, relevances) - 1
    else:
        raise ValueError("variant must be either 'linear' or 'exponent'")

    return float(np.sum(gains * discounts))


def compute_ndcg_at_k(
    relevances: Sequence[float],
    k: int = 10,
    variant: Literal["linear", "exponent"] = "linear",
) -> float:
    dcg = dcg_at_k(relevances, k, variant)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k, variant)
    return 0.0 if idcg == 0 else dcg / idcg
