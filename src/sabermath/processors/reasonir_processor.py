from typing import ClassVar

import numpy as np

from .embedding_processor import EmbeddingProcessor

DEFAULT_MODEL = "reasonir/ReasonIR-8B"


class ReasonIRProcessor(EmbeddingProcessor):
    processor: ClassVar[str | None] = "reasonir"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        instruction: str = "",
        encode_batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._instruction = instruction
        self._encode_batch_size = encode_batch_size
        self._model = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype="auto",  # activates bf16 per the model card
                trust_remote_code=True,  # custom bidirectional encoder
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            self._model = model
        return self._model

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        instruction: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        model = self._get_model()
        instr = self._instruction if instruction is None else instruction

        try:
            embs = model.encode(
                texts, instruction=instr, batch_size=self._encode_batch_size
            )
        except TypeError:
            # Fall back to the minimal official call signature if the remote
            # code does not accept the batch_size keyword.
            embs = model.encode(texts, instruction=instr)

        if hasattr(embs, "detach"):  # torch tensor -> numpy
            embs = embs.detach().float().cpu().numpy()
        return np.atleast_2d(np.asarray(embs, dtype=np.float32))
