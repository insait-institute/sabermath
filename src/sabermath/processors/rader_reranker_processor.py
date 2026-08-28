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
- THE INPUT TEMPLATE. rerank.py builds f"query: {q} document: {d}{eos}";
  this builds "query: Query: {q} document:  {d} <eos>" (the "T10" arm of
  experiments/rader-reranker-diag/sweep.py). Two independent reasons, each
  measured on a frozen 60-query subset (statement-statement, nDCG@10):
    * a space before the eos: +0.0167 [+0.0072,+0.0262], p=0.001,
      replicated across two independent jobs. The paper's Appendix G writes
      the input as "query:{q} document:{d} <eos>" - that space is real.
      (Appendix G ALSO drops the post-colon spaces; that part is neutral to
      harmful, -0.0046 [-0.0127,+0.0036], so it is NOT copied here.)
    * the Tevatron reconstruction ("Query: " prefix, which every training
      row carries, + the empty-title double space): +0.0064, p=0.31 alone.
  Together +0.0223 [+0.0089,+0.0357], p=0.002, and additive (interaction
  -0.0008). NOTE the combination is not significantly better than the
  pre-eos space alone (+0.0056, p=0.35) - it is adopted because it also
  matches the recovered training format, not on its own evidence.
  Every published rader-reranker-7b number PREDATES this template.
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
        """Build "query: Query: {q} document:  {d} <eos>" - the T10 template
        (see the module docstring for why it, and not rerank.py's).

        Only the document is truncated, and the trailing space + eos are
        appended AFTER truncating, so the scoring position is always the eos
        and the pre-eos space can never be the thing that gets trimmed.

        Two spacing details are load-bearing and must not be "tidied":
        - The space between "document:" and the document rides on the
          DOCUMENT side (prefix ends "document: ", doc is " " + document).
          Qwen2 BPE merges a leading space into the following token
          (" Problem" is ONE token), so splitting the pair anywhere else
          changes the ids. Verified: this split reproduces the one-shot
          tokenization of the T10 string on 625/625 real pairs.
        - The double space after "document:" is what Tevatron's format_pair
          actually produced for this checkpoint (empty title slot), and
          " " vs "  " are DIFFERENT single tokens (220 vs 256), so the
          doubling is a real change, not whitespace noise."""
        prefix_ids = self._tokenizer(
            f"query: Query: {query} document: ", add_special_tokens=False
        ).input_ids
        doc_ids = self._tokenizer(" " + document, add_special_tokens=False).input_ids
        tail_ids = self._tokenizer(" ", add_special_tokens=False).input_ids

        budget = self._max_length - len(prefix_ids) - len(tail_ids) - 1
        if budget > 0 and len(doc_ids) > budget:
            doc_ids = doc_ids[:budget]

        return prefix_ids + doc_ids + tail_ids + [self._tokenizer.eos_token_id]

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
