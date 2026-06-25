import asyncio
import os
import warnings
from collections.abc import Iterator

import numpy as np
from tqdm.asyncio import tqdm

from .embedding_processor import EmbeddingProcessor


class OpenAIProcessor(EmbeddingProcessor):
    processor = "openai"

    def __init__(self, model_name: str, api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Please install openai to use it as a processor") from e

        client_args = {}

        if api_key is not None:
            client_args["api_key"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            warnings.warn(
                "No OpenAI API key was provided. Set OPENAI_API_KEY or pass "
                "api_key=... explicitly. This may cause authentication issues.",
                stacklevel=2,
            )

        self._model_name = model_name
        self._client = AsyncOpenAI(**client_args)

    @property
    def model(self) -> str:
        return self._model_name

    def _split_to_batches(
        self,
        items: list[str],
        batch_size: int,
    ) -> Iterator[list[str]]:
        if batch_size <= 0:
            raise ValueError('"batch_size" must be >= 1')

        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    async def _encode_one(
        self,
        text: str,
        sem: asyncio.Semaphore,
        *,
        retries: int = 4,
        **kwargs: object,
    ) -> list[float]:
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                async with sem:
                    response = await self._client.embeddings.create(
                        model=self._model_name,
                        input=text,
                        **kwargs,
                    )

                return response.data[0].embedding

            except Exception as e:
                last_error = e

                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(
            f"Failed to encode text after {retries} attempts."
        ) from last_error

    async def encode_async(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        max_concurrency: int = 20,
        retries: int = 3,
        **kwargs: object,
    ) -> np.ndarray:
        if max_concurrency <= 0:
            raise ValueError('"max_concurrency" must be >= 1')

        sem = asyncio.Semaphore(max_concurrency)

        coros = [
            self._encode_one(
                text,
                sem,
                retries=retries,
                **kwargs,
            )
            for text in texts
        ]

        if show_progress_bar:
            results = await tqdm.gather(*coros)
        else:
            results = await asyncio.gather(*coros)

        return np.asarray(results, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        retries: int = 9,
        max_concurrency: int = 20,
        **kwargs: object,
    ) -> np.ndarray:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.encode_async(
                    texts,
                    show_progress_bar=show_progress_bar,
                    retries=retries,
                    max_concurrency=max_concurrency,
                    **kwargs,
                )
            )

        raise RuntimeError(
            ".encode() can only be called from a synchronous context. "
            "Use .encode_async() inside async code."
        )
