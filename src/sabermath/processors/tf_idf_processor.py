from typing import ClassVar

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import ModelProcessor

from .tokenization_helper import math_word_tokens

class TfidfProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "tf-idf"

    def __init__(self, tokenize_approach0: bool = True, **vectorizer_kwargs) -> None:
        self.tokenize_approach0 = tokenize_approach0
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
        score_batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []

        if self.tokenize_approach0:
            vectorizer = TfidfVectorizer(
                tokenizer=math_word_tokens,
                token_pattern=None,
                lowercase=False,
                preprocessor=None,
                **self.vectorizer_kwargs,
                **kwargs,
            )
        else:
            # Matches the old TfidfProcessor: no custom tokenizer at all, so
            # sklearn's own defaults (regex tokenization + lowercasing) apply.
            vectorizer = TfidfVectorizer(**self.vectorizer_kwargs, **kwargs)

        doc_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])

        if score_batch_size is None:
            return cosine_similarity(query_vector, doc_vectors)[0].tolist()

        # score_batch_size: similarity computed in fixed-size document slices
        # while the vectorizer stays FIT ON THE FULL candidate set (the IDF
        # statistics must never be chunked - that would change the scores).
        # Exists purely for the timing harness's uniform 16-documents-per-step
        # protocol; the underlying op is CPU-vectorized either way.
        scores: list[float] = []
        for start in range(0, doc_vectors.shape[0], score_batch_size):
            block = cosine_similarity(
                query_vector, doc_vectors[start : start + score_batch_size]
            )[0]
            scores.extend(float(s) for s in block)
        return scores
