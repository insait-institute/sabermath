import pickle
from abc import abstractmethod
from itertools import chain

from datasets import Dataset
import numpy as np

from .base import ModelProcessor


class EmbeddingProcessor(ModelProcessor):
    @classmethod
    def from_huggingface(cls, model_name: str):
        raise NotImplemented

    def _cosine_similarity(
        self, query_emb: np.ndarray, docs_embs: np.ndarray
    ) -> np.ndarray:
        q_norm = np.linalg.norm(query_emb)
        d_norms = np.linalg.norm(docs_embs, axis=1, keepdims=True)

        if q_norm == 0:
            raise ValueError("Query embedding has zero norm")

        if np.any(d_norms == 0):
            raise ValueError("At least one document embedding has zero norm")

        query_emb = query_emb / q_norm
        docs_embs = docs_embs / d_norms
        return docs_embs @ query_emb

    def export_cache(self, path: str) -> None:
        if not hasattr(self, "_vector_cache"):
            raise ValueError("No cache to export")

        with open(path, "wb") as f:
            pickle.dump(self._vector_cache, f)

    def import_cache(self, path_or_data: str, is_path: bool = True) -> None:
        if is_path:
            with open(path_or_data, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = path_or_data

        if not hasattr(self, "_vector_cache"):
            self._vector_cache = cache
        else:
            self._vector_cache.update(cache)

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        check_cache: bool = True,
        update_cache: bool = True,
        **kwargs,
    ) -> list[float]:
        if not hasattr(self, "_vector_cache"):
            self._vector_cache = {}

        N_d = len(documents)

        if check_cache:
            embeddings = [None for _ in range(N_d + 1)]
            encode_texts: list[str] = []
            idx_map: list[int] = []

            for i, doc in enumerate(chain(documents, (query,))):
                if doc in self._vector_cache:
                    embeddings[i] = self._vector_cache[doc]
                else:
                    encode_texts.append(doc)
                    idx_map.append(i)

            if encode_texts:
                new_emb = self.encode(
                    encode_texts, show_progress_bar=show_progress_bar, **kwargs
                )

                if update_cache:
                    for text, emb in zip(encode_texts, new_emb):
                        self._vector_cache[text] = emb

                for idx, emb in zip(idx_map, new_emb):
                    embeddings[idx] = emb

            query_embedding = embeddings[N_d]
            document_embeddings = embeddings[:N_d]

        else:
            encode_texts = documents + [query]
            embeddings = self.encode(
                encode_texts, show_progress_bar=show_progress_bar, **kwargs
            )

            query_embedding = embeddings[N_d]
            document_embeddings = embeddings[:N_d]

        scores = self._cosine_similarity(query_embedding, document_embeddings)

        return scores

    @abstractmethod
    def encode(
        self, texts: list[str], show_progress_bar: bool = True, **kwargs
    ) -> np.ndarray:
        """Encode a list of texts into a list of vectors."""
        pass

    def encode_statements(
        self,
        ds: Dataset,
        show_progress_bar: bool = True,
        *,
        statement_column: str = "problem",
        **kwargs,
    ) -> np.ndarray:
        statements = list(ds[statement_column])

        return self.encode(
            statements,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

    def encode_full(
        self,
        ds: Dataset,
        show_progress_bar: bool = True,
        *,
        statement_column: str = "problem",
        solution_column: str = "solution",
        **kwargs,
    ) -> np.ndarray:
        statements = list(ds[statement_column])
        solutions = list(ds[solution_column])

        full_texts = [
            f"Problem: {statement}\n\nSolution: {solution}"
            for statement, solution in zip(statements, solutions)
        ]

        return self.encode(full_texts, show_progress_bar=show_progress_bar, **kwargs)
