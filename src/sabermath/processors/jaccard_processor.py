from typing import ClassVar

from .base import ModelProcessor

from .tokenization_helper import math_word_tokens

class JaccardProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "jaccard"

    def __init__(self, lowercase: bool = True, tokenize_approach0: bool = True) -> None:
        self.lowercase = lowercase
        self.tokenize_approach0 = tokenize_approach0

    @property
    def model(self) -> str:
        return "jaccard"

    def _tokenize(self, text: str) -> set[str]:
        if self.tokenize_approach0:
            return set(math_word_tokens(text, lowercase=self.lowercase))

        if self.lowercase:
            text = text.lower()

        return set(text.split())

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float]:
        query_tokens = self._tokenize(query)

        scores: list[float] = []

        for document in documents:
            document_tokens = self._tokenize(document)

            union = query_tokens | document_tokens
            if not union:
                scores.append(0.0)
                continue

            intersection = query_tokens & document_tokens
            scores.append(len(intersection) / len(union))

        return scores
