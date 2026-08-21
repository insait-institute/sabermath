"""naver/splade-code-8B: SPLADE sparse retrieval (Qwen3-based), run via
Sentence Transformers' SparseEncoder exactly as in the official Hugging Face
model card:

    from sentence_transformers import SparseEncoder
    model = SparseEncoder("naver/splade-code-8B", trust_remote_code=True)

Queries and documents are encoded with encode_query()/encode_document() (the
SparseEncoder counterparts of the model card's `prompt_type="query"`
Transformers usage) and scored with the model's native similarity function
(dot product for sparse embeddings, not cosine) - this is why SpladeProcessor
implements get_scores() directly instead of subclassing EmbeddingProcessor.

Ported from
rag-math-test/rank-embedding-math/running-rerankers/sabermath_splade.py.
"""

from typing import ClassVar

import numpy as np

from .base import ModelProcessor

DEFAULT_MODEL = "naver/splade-code-8B"


class SpladeProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "splade"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        query_batch_size: int = 1,
        document_batch_size: int = 4,
        model_kwargs: dict | None = None,
    ) -> None:
        self._model_name = model_name
        self._query_batch_size = query_batch_size
        self._document_batch_size = document_batch_size
        # torch_dtype bfloat16 by default (2026-08-21): BOTH splade-code
        # checkpoints DECLARE torch_dtype bfloat16 in their configs, but a
        # bare SparseEncoder(...) load ignores that and computes in fp32 -
        # the only fp32 compute left in the benchmark alongside the (then)
        # rader/pylate paths. Validated on the frozen feasibility sample:
        # fp32-vs-bf16 rankings are IDENTICAL (Spearman 1.0000, |dNDCG@10|
        # 0.0000 for both sizes) while the 8B scores ~28x faster.
        self._model_kwargs = (
            dict(model_kwargs)
            if model_kwargs is not None
            else {"torch_dtype": "bfloat16"}
        )
        self._model = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SparseEncoder

            self._model = SparseEncoder(
                self._model_name,
                trust_remote_code=True,
                model_kwargs=self._model_kwargs,
            )
        return self._model

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []

        model = self._get_model()

        query_embedding = model.encode_query(
            [query], batch_size=self._query_batch_size
        )
        document_embeddings = model.encode_document(
            documents, batch_size=self._document_batch_size
        )

        scores = model.similarity(query_embedding, document_embeddings)
        if hasattr(scores, "detach"):  # torch.Tensor -> numpy (handles GPU/bf16)
            scores = scores.detach().cpu().float().numpy()
        return np.asarray(scores).ravel().tolist()
