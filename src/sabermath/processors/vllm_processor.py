import numpy as np

from .embedding_processor import EmbeddingProcessor


def _get_model_name(llm) -> str | None:
    return getattr(
        getattr(getattr(llm, "llm_engine", None), "model_config", None), "model", None
    ) or getattr(getattr(llm, "model_config", None), "model", None)


class VLLMProcessor(EmbeddingProcessor):
    processor = "vllm"

    def __init__(
        self,
        llm: "vllm.LLM",
        model_name: str | None = None,
    ):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("Please install vllm to use it as a processor") from e

        if not isinstance(llm, LLM):
            raise TypeError(
                "VLLMProcessor can only be build using a valid vllm.LLM object"
            )

        self._model_name = model_name or _get_model_name(llm)
        self._llm = llm

    @classmethod
    def from_huggingface(cls, model_name: str, **kwargs) -> "VLLMProcessor":
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("Please install vllm to use it as a processor") from e

        llm = LLM(
            model=model_name,
            runner="pooling",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
            **kwargs,
        )
        return cls(llm, model_name)

    @property
    def model(self) -> str | None:
        return self._model_name

    def encode(
        self, texts: list[str], show_progress_bar: bool = True, **kwargs
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)

        outputs = self._llm.embed(texts, use_tqdm=show_progress_bar, **kwargs)

        return np.array([o.outputs.embedding for o in outputs])
