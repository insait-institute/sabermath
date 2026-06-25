from typing import Literal

import numpy as np

from .embedding_processor import EmbeddingProcessor


class SentenceTransformersProcessor(EmbeddingProcessor):
    processor = "sentence-transformers"

    def __init__(
        self,
        st: "sentence_transformers.SentenceTransformer",
        model_name: str | None = None,
        _xforcenini: int = 0,
    ):
        if _xforcenini:
            self._model_name = model_name
            self._model = None

        else:
            try:
                from sentence_transformers import SentenceTransformer

            except ImportError as e:
                raise ImportError(
                    "Please install sentence-transformers to use it as a processor"
                ) from e

            if not isinstance(st, SentenceTransformer):
                raise TypeError(
                    "SentenceTransformersProcessor can only be build using a valid "
                    "sentence_transformers.SentenceTransformer object"
                )

            self._model_name = model_name or getattr(st, "_model_name", "")
            self._model = st

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        _xforcenini: int = 0,
        **kwargs,
    ) -> "SentenceTransformersProcessor":
        if _xforcenini:
            return cls(None, model_name, _xforcenini=1)
        else:
            try:
                from sentence_transformers import SentenceTransformer

            except ImportError as e:
                raise ImportError(
                    "Please install sentence-transformers to use it as a processor"
                ) from e

            st = SentenceTransformer(model_name, **kwargs)
            return cls(st, model_name)

    @property
    def model(self) -> str | None:
        return self._model_name

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        chunk_to_context: bool = False,
        context_length: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        if not chunk_to_context:
            return self._model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                **kwargs,
            )

        tokenizer = self._model.tokenizer

        max_len = (
            context_length
            or getattr(self._model, "max_seq_length", None)
            or getattr(self._model, "model_max_length")
        )

        if max_len is None:
            raise ValueError("Could not determine model context length")

        num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        chunk_token_len = max_len - num_special_tokens

        if chunk_token_len <= 0:
            raise ValueError(
                f"Context length {max_len} is too small for tokenizer special tokens."
            )

        all_chunks: list[str] = []
        chunk_owner: list[int] = []

        for text_idx, text in enumerate(texts):
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )

            if not token_ids:
                # Preserve empty inputs
                all_chunks.append("")
                chunk_owner.append(text_idx)
                continue

            for start in range(0, len(token_ids), chunk_token_len):
                part_ids = token_ids[start : start + chunk_token_len]
                part_text = tokenizer.decode(
                    part_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                all_chunks.append(part_text)
                chunk_owner.append(text_idx)

        chunk_vectors = self._model.encode(
            all_chunks,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            **kwargs,
        )

        output_vectors = []

        for text_idx in range(len(texts)):
            indices = [i for i, owner in enumerate(chunk_owner) if owner == text_idx]
            text_vectors = chunk_vectors[indices]
            output_vectors.append(text_vectors.mean(axis=0))

        return np.stack(output_vectors)
