import asyncio
from collections.abc import Iterator
import math
import os
from typing import Any
import warnings

import numpy as np
from tqdm.asyncio import tqdm

from .embedding_processor import EmbeddingProcessor


class GoogleProcessor(EmbeddingProcessor):
    processor = "google"

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Please install google-genai to use it as a processor"
            ) from e

        client_args: dict[str, Any] = {}

        if base_url is None:
            base_url = os.getenv("GEMINI_BASE_URL")

        if base_url:
            # client_args["vertexai"] = True
            client_args["http_options"] = {
                "base_url": base_url,
                "headers": headers or {},
            }

        if api_key:
            client_args["api_key"] = api_key
        elif not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            warnings.warn(
                "No Gemini API key was provided. Set GEMINI_API_KEY or "
                "GOOGLE_API_KEY, or pass api_key=... explicitly. This may "
                "cause authentication issues.",
                stacklevel=2,
            )

        self._model_name = model_name
        self._client = genai.Client(**client_args)

    @property
    def model(self) -> str:
        return self._model_name

    def _split_to_batches(
        self, items: list[str], batch_size: int
    ) -> Iterator[list[str]]:
        if batch_size <= 0:
            raise ValueError('"batch_size" must be >= 1')

        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _concat(self, batches: list[list[list[float]]]) -> list[list[float]]:
        return [item for batch in batches for item in batch]

    async def _encode_batch(
        self,
        texts: list[str],
        sem: asyncio.Semaphore,
        *,
        retries: int = 9,
    ) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                async with sem:
                    response = await self._client.aio.models.embed_content(
                        model=self._model_name,
                        contents=texts,
                    )

                    if response.embeddings is None:
                        raise RuntimeError("Gemini returned no embeddings.")

                    embeddings = [embedding.values for embedding in response.embeddings]

                    return embeddings

            except Exception as e:
                last_error = e

                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(
            f"Failed to encode batch after {retries} attempts."
        ) from last_error

    async def encode_async(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        retries: int = 9,
        batch_size: int = 100,
        max_concurrency: int = 20,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._model_name.startswith("gemini-embedding-2"):
            batch_size = 1
            max_concurrency = math.ceil(max_concurrency * 1.2)

        sem = asyncio.Semaphore(max_concurrency)
        batches = list(self._split_to_batches(texts, batch_size))

        coros = [
            self._encode_batch(batch, sem, retries=retries, **kwargs)
            for batch in batches
        ]

        if show_progress_bar:
            batched_results = await tqdm.gather(*coros)
        else:
            batched_results = await asyncio.gather(*coros)

        results = self._concat(batched_results)

        return np.asarray(results, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        retries: int = 9,
        batch_size: int = 100,
        max_concurrency: int = 20,
        **kwargs: Any,
    ) -> np.ndarray:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.encode_async(
                    texts,
                    show_progress_bar=show_progress_bar,
                    retries=retries,
                    batch_size=batch_size,
                    max_concurrency=max_concurrency,
                    **kwargs,
                )
            )

        raise RuntimeError(
            ".encode() can only be called from a synchronous context. "
            "Use .encode_async() inside async code."
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    def close(self) -> None:
        self._client.close()
