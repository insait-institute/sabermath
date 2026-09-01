
from typing import ClassVar

from .base import ModelProcessor

from .tokenization_helper import math_word_tokens

class BM25Processor(ModelProcessor):
    processor: ClassVar[str | None] = "bm25"

    def __init__(
        self,
        *,
        lowercase: bool = True,
        tokenize_approach0: bool = True,
        **bm25_kwargs,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._BM25Okapi = BM25Okapi
        except ImportError as e:
            raise ImportError(
                "Please install rank_bm25 to use BM25 as a processor"
            ) from e
        self.lowercase = lowercase
        self.tokenize_approach0 = tokenize_approach0
        self.bm25_kwargs = bm25_kwargs

    @property
    def model(self) -> str:
        return "bm25"

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenize_approach0:
            return math_word_tokens(text, lowercase=self.lowercase)

        if self.lowercase:
            text = text.lower()

        return text.split()

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        score_batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []

        tokenized_documents = [self._tokenize(document) for document in documents]
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return [0.0 for _ in documents]

        bm25 = self._BM25Okapi(
            tokenized_documents,
            **self.bm25_kwargs,
            **kwargs,
        )

        if score_batch_size is None:
            return bm25.get_scores(tokenized_query).tolist()

        # score_batch_size: score in fixed-size document slices AGAINST THE
        # FULL-CORPUS index (get_batch_scores uses the same corpus statistics
        # as get_scores, so the numbers are identical) - exists purely so the
        # timing harness can impose its uniform 16-documents-per-step
        # protocol. Scoring here is CPU-vectorized either way; this caps, not
        # creates, parallelism.
        scores: list[float] = []
        for start in range(0, len(documents), score_batch_size):
            doc_ids = list(range(start, min(start + score_batch_size, len(documents))))
            scores.extend(float(s) for s in bm25.get_batch_scores(tokenized_query, doc_ids))
        return scores
