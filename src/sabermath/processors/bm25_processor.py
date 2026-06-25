from typing import ClassVar

from .base import ModelProcessor


class BM25Processor(ModelProcessor):
    processor: ClassVar[str | None] = "bm25"

    def __init__(self, *, lowercase: bool = True, **bm25_kwargs) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._BM25Okapi = BM25Okapi
        except ImportError as e:
            raise ImportError(
                "Please install rank_bm25 to use BM25 as a processor"
            ) from e
        self.lowercase = lowercase
        self.bm25_kwargs = bm25_kwargs

    @property
    def model(self) -> str:
        return "bm25"

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()

        return text.split()

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

        tokenized_documents = [self._tokenize(document) for document in documents]
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return [0.0 for _ in documents]

        bm25 = self._BM25Okapi(
            tokenized_documents,
            **self.bm25_kwargs,
            **kwargs,
        )

        scores = bm25.get_scores(tokenized_query)

        return scores.tolist()
