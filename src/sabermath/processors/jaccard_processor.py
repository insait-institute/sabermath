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
        score_batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        query_tokens = self._tokenize(query)

        # score_batch_size only groups the (already sequential, per-document)
        # loop into fixed-size slices - accepted for the timing harness's
        # uniform 16-documents-per-step protocol, but it neither changes the
        # scores nor creates any real parallelism here.
        step = score_batch_size or len(documents) or 1

        scores: list[float] = []
        for start in range(0, len(documents), step):
            for document in documents[start : start + step]:
                document_tokens = self._tokenize(document)

                union = query_tokens | document_tokens
                if not union:
                    scores.append(0.0)
                    continue

                intersection = query_tokens & document_tokens
                scores.append(len(intersection) / len(union))

        return scores
