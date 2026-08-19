"""Raderspace/reranker_Qwen25_7B_NuminaMath_MATH_allquerytypes: RaDeR's
pointwise cross-encoder reranker.

The HF repo ships only a LoRA adapter (adapter_config.json +
adapter_model.safetensors; task_type=SEQ_CLS with the "score" head in
modules_to_save) on top of Qwen/Qwen2.5-7B-Instruct. Loading and scoring
follow the reference implementation in the RaDeR GitHub repo (rerank.py's
`get_model_new` + `Reranker_models.rerank` at
https://github.com/Debrup-61/RaDeR):

    base = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", num_labels=1, torch_dtype=bfloat16)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    input = f"query: {query} document: {doc}{eos_token}"
    score = model(**tokenize(input)).logits[0][0]

(The base-model load warns that the fresh `score` head is uninitialized -
expected: the trained head arrives with the adapter via modules_to_save.)

Deviations from the reference, all deliberate:
- flash_attention_2 is opt-in (`use_flash_attn=True`) rather than hardcoded,
  since flash-attn is a separate compiled install not present in this
  pipeline's envs; SDPA attention is numerically equivalent.
- Documents are token-truncated so each sequence fits `max_length` (the
  reference applies no truncation at all, which is fine for BRIGHT's short
  passages but not guaranteed for SABER-Math's problem+solution documents).
  The eos token is appended *after* truncation so the scoring position is
  always the eos, exactly as in the reference format.
- Pairs are scored in padded batches instead of one-by-one (the reference
  loops with batch size 1). Qwen2ForSequenceClassification locates each
  row's last non-pad token for pooling, so right-padding does not change the
  scores, and Qwen2.5's pad token (<|endoftext|>) differs from its eos
  (<|im_end|>), so the appended eos is never mistaken for padding.
"""

from typing import ClassVar

from .base import ModelProcessor

DEFAULT_LORA_MODEL = "Raderspace/reranker_Qwen25_7B_NuminaMath_MATH_allquerytypes"


class RaDeRRerankerProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "rader-reranker"

    def __init__(
        self,
        lora_model_name: str = DEFAULT_LORA_MODEL,
        *,
        base_model_name: str | None = None,
        batch_size: int = 4,
        max_length: int = 8192,
        use_flash_attn: bool = False,
    ) -> None:
        self._lora_model_name = lora_model_name
        self._base_model_name = base_model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_flash_attn = use_flash_attn

        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> str | None:
        return self._lora_model_name

    def _init(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            from peft import PeftConfig, PeftModel
        except ImportError as e:
            raise ImportError(
                "Please install peft to use RaDeRRerankerProcessor"
            ) from e

        base_name = self._base_model_name
        if base_name is None:
            base_name = PeftConfig.from_pretrained(
                self._lora_model_name
            ).base_model_name_or_path

        self._tokenizer = AutoTokenizer.from_pretrained(base_name)

        model_kwargs = {
            "num_labels": 1,
            "torch_dtype": (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
        }
        if self._use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name, **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, self._lora_model_name)
        model = model.merge_and_unload()

        if model.config.pad_token_id is None:
            model.config.pad_token_id = self._tokenizer.pad_token_id

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model

    def _build_input_ids(self, query: str, document: str) -> list[int]:
        """Tokenize "query: {q} document: {d}", truncate the document part so
        the sequence (plus the final eos) fits max_length, then append eos."""
        prefix_ids = self._tokenizer(
            f"query: {query} document: ", add_special_tokens=False
        ).input_ids
        doc_ids = self._tokenizer(document, add_special_tokens=False).input_ids

        budget = self._max_length - len(prefix_ids) - 1
        if budget > 0 and len(doc_ids) > budget:
            doc_ids = doc_ids[:budget]

        return prefix_ids + doc_ids + [self._tokenizer.eos_token_id]

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

        self._init()

        import torch

        all_input_ids = [self._build_input_ids(query, doc) for doc in documents]

        scores: list[float] = []
        for start in range(0, len(all_input_ids), self._batch_size):
            batch = all_input_ids[start : start + self._batch_size]
            inputs = self._tokenizer.pad(
                {"input_ids": batch}, padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            scores.extend(logits[:, 0].float().cpu().tolist())

        return scores
