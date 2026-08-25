"""PyLate ColBERT late-interaction rerankers, e.g. lightonai/GTE-ModernColBERT-v1
and lightonai/Reason-ModernColBERT.

SABER-Math is a reranking task over a preselected ~150-document candidate set
per query, so this uses the official "Reranking" usage from the Hugging Face
model card (`pylate.rank.rerank` on encoded queries/documents, MaxSim
scoring), without building a Voyager index. Requires `pip install -U pylate`.

Ported from
rag-math-test/rank-embedding-math/running-rerankers/sabermath_gte_colbert.py
and sabermath_reason_colbert.py (which differed only in MODEL_NAME - this
Processor is parameterized by model_name so it covers both).
"""

from typing import ClassVar

from .base import ModelProcessor


class ColBERTProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "pylate-colbert"

    def __init__(
        self,
        model_name: str,
        *,
        encode_batch_size: int = 32,
        model_kwargs: dict | None = None,
        query_length: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._encode_batch_size = encode_batch_size
        self._query_length = query_length
        # torch_dtype bfloat16 by default (2026-08-21, precision-fairness
        # rollout: no model computes in fp32 anymore). NOTE this one is a
        # deliberate deviation from the checkpoints (both ModernColBERT
        # configs declare fp32): validated on the frozen feasibility sample
        # at Spearman 0.9993 (reason-moderncolbert) and 0.9907
        # (gte-moderncolbert, NEAR-MATCH band - same acceptance category as
        # reasonir/jina-v5-small), with ~10-35x faster scoring. Pass
        # model_kwargs={"torch_dtype": "float32"} to reproduce the published
        # fp32 numbers exactly.
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
            try:
                from pylate import models
            except ImportError as e:
                raise ImportError(
                    "Please install pylate to use ColBERTProcessor"
                ) from e
            colbert_kwargs = {}
            if self._query_length is not None:
                colbert_kwargs["query_length"] = self._query_length
            self._model = models.ColBERT(
                model_name_or_path=self._model_name,
                model_kwargs=self._model_kwargs,
                **colbert_kwargs,
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

        from pylate import rank

        model = self._get_model()

        # PyLate's rerank API is batched over queries; wrap the single query
        # and its candidate list, per the model-card snippet.
        queries_embeddings = model.encode(
            [query],
            batch_size=self._encode_batch_size,
            is_query=True,
            show_progress_bar=show_progress_bar,
        )
        documents_embeddings = model.encode(
            [documents],
            batch_size=self._encode_batch_size,
            is_query=False,
            show_progress_bar=show_progress_bar,
        )

        documents_ids = [list(range(len(documents)))]
        reranked = rank.rerank(
            documents_ids=documents_ids,
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )[0]

        # rank.rerank returns, per query, candidates as dicts with "id" (our
        # local index) and "score". Safety net: if PyLate ever drops a
        # candidate (it should not), it keeps a default score of 0.0 rather
        # than crashing - benign since MaxSim scores are non-negative, so a
        # dropped candidate just sinks to the bottom of the ranking.
        scores = [0.0] * len(documents)
        for entry in reranked:
            scores[int(entry["id"])] = float(entry["score"])
        return scores
