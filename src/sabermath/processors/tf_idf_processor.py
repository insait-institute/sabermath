from typing import ClassVar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import ModelProcessor


class TfidfProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "tf-idf"

    def __init__(self, **vectorizer_kwargs) -> None:
        self.vectorizer_kwargs = vectorizer_kwargs

    @property
    def model(self) -> str:
        return "tf-idf"

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

        vectorizer = TfidfVectorizer(
            **self.vectorizer_kwargs,
            **kwargs,
        )

        doc_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(query_vector, doc_vectors)[0]

        return scores.tolist()
