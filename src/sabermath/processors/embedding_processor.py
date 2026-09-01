import pickle
from abc import abstractmethod
from itertools import chain

from datasets import Dataset
import numpy as np

from .base import ModelProcessor

AFFIX_KEYS = (
    "query_prompt",
    "document_prompt",
    "query_suffix",
    "document_suffix",
    "query_encode_kwargs",
    "document_encode_kwargs",
)


def split_affix_kwargs(kwargs: dict) -> tuple[dict, dict]:
    affixes = {k: v for k, v in kwargs.items() if k in AFFIX_KEYS}
    rest = {k: v for k, v in kwargs.items() if k not in AFFIX_KEYS}
    return affixes, rest


def apply_affixes(
    texts: list[str], prompt: str | None, suffix: str | None
) -> list[str]:
    if not prompt and not suffix:
        return list(texts)
    head = prompt or ""
    tail = suffix or ""
    return [f"{head}{text}{tail}" for text in texts]


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

    def _encode_side(
        self,
        texts: list[str],
        *,
        tag: str | None,
        show_progress_bar: bool,
        check_cache: bool,
        update_cache: bool,
        encode_kwargs: dict,
    ) -> list:
        def key(text: str):
            return text if tag is None else (tag, text)

        embeddings: list = [None for _ in texts]
        encode_texts: list[str] = []
        idx_map: list[int] = []

        for i, text in enumerate(texts):
            if check_cache and key(text) in self._vector_cache:
                embeddings[i] = self._vector_cache[key(text)]
            else:
                encode_texts.append(text)
                idx_map.append(i)

        if encode_texts:
            new_emb = self.encode(
                encode_texts,
                show_progress_bar=show_progress_bar,
                **encode_kwargs,
            )
            if check_cache and update_cache:
                for text, emb in zip(encode_texts, new_emb):
                    self._vector_cache[key(text)] = emb
            for idx, emb in zip(idx_map, new_emb):
                embeddings[idx] = emb

        return embeddings

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        check_cache: bool = True,
        update_cache: bool = True,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        query_suffix: str | None = None,
        document_suffix: str | None = None,
        query_encode_kwargs: dict | None = None,
        document_encode_kwargs: dict | None = None,
        **kwargs,
    ) -> list[float]:
        if not hasattr(self, "_vector_cache"):
            self._vector_cache = {}

        query = apply_affixes([query], query_prompt, query_suffix)[0]
        documents = apply_affixes(documents, document_prompt, document_suffix)

        if query_encode_kwargs or document_encode_kwargs:
            return self._get_scores_per_side(
                query,
                documents,
                show_progress_bar=show_progress_bar,
                check_cache=check_cache,
                update_cache=update_cache,
                query_encode_kwargs=query_encode_kwargs or {},
                document_encode_kwargs=document_encode_kwargs or {},
                **kwargs,
            )

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

    def _get_scores_per_side(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool,
        check_cache: bool,
        update_cache: bool,
        query_encode_kwargs: dict,
        document_encode_kwargs: dict,
        **kwargs,
    ) -> list[float]:
        def side_tag(side_kwargs: dict) -> str | None:
            if not side_kwargs:
                return None
            return "|".join(f"{k}={side_kwargs[k]}" for k in sorted(side_kwargs))

        document_embeddings = self._encode_side(
            documents,
            tag=side_tag(document_encode_kwargs),
            show_progress_bar=show_progress_bar,
            check_cache=check_cache,
            update_cache=update_cache,
            encode_kwargs={**kwargs, **document_encode_kwargs},
        )
        query_embedding = self._encode_side(
            [query],
            tag=side_tag(query_encode_kwargs),
            show_progress_bar=show_progress_bar,
            check_cache=check_cache,
            update_cache=update_cache,
            encode_kwargs={**kwargs, **query_encode_kwargs},
        )[0]

        return self._cosine_similarity(
            np.asarray(query_embedding), np.asarray(document_embeddings)
        )

    @abstractmethod
    def encode(
        self, texts: list[str], show_progress_bar: bool = True, **kwargs
    ) -> np.ndarray:
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
