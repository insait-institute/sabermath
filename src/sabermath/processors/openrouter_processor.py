import asyncio
import os
import warnings

import numpy as np

from .embedding_processor import EmbeddingProcessor


class OpenRouterEmbeddingProcessor(EmbeddingProcessor):
    """text-embedding-3-small/large via OpenRouter instead of OpenAI
    directly - requested explicitly (OpenRouter key supplied instead of an
    OpenAI one). sabermath.processors.OpenAIProcessor has no way to point
    at a custom base_url; this replicates its async batching/retry logic
    with a different base_url and the OpenRouter model-id convention
    (provider-prefixed, e.g. "openai/text-embedding-3-small" - confirmed
    directly: OpenRouter exposes an OpenAI-API-compatible /v1/embeddings
    endpoint, verified with a real embeddings.create() call before wiring
    this in)."""

    processor = "openrouter"

    def __init__(self, model_name: str, api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Please install openai to use it as a processor") from e

        client_args = {"base_url": "https://openrouter.ai/api/v1"}

        if api_key is not None:
            client_args["api_key"] = api_key
        elif os.getenv("OPENROUTER_API_KEY"):
            client_args["api_key"] = os.getenv("OPENROUTER_API_KEY")
        else:
            warnings.warn(
                "No OpenRouter API key was provided. Set OPENROUTER_API_KEY "
                "or pass api_key=... explicitly. This may cause "
                "authentication issues.",
                stacklevel=2,
            )

        self._model_name = model_name
        self._client = AsyncOpenAI(**client_args)

    @property
    def model(self) -> str:
        return self._model_name

    async def _encode_one(self, text, sem, *, retries: int = 4, **kwargs):
        last_error = None
        for attempt in range(retries):
            try:
                async with sem:
                    response = await self._client.embeddings.create(
                        model=f"openai/{self._model_name}", input=text, **kwargs
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
        **kwargs,
    ) -> np.ndarray:
        if max_concurrency <= 0:
            raise ValueError('"max_concurrency" must be >= 1')
        sem = asyncio.Semaphore(max_concurrency)
        coros = [self._encode_one(text, sem, retries=retries, **kwargs) for text in texts]
        results = await asyncio.gather(*coros)
        return np.asarray(results, dtype=np.float32)

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        retries: int = 9,
        max_concurrency: int = 20,
        **kwargs,
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
            ".encode() can only be called from a synchronous context."
        )
