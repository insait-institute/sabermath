"""Qwen/Qwen3-Reranker-8B: a cross-encoder reranker.

Each (query, document) pair is formatted with the <Instruct>/<Query>/
<Document> template, wrapped in the yes/no judging chat template, and scored
as P("yes") from the last-token logits, following the officially supplied
Transformers usage from the Hugging Face model card
(https://huggingface.co/Qwen/Qwen3-Reranker-8B). Requires transformers>=4.51.0.

Ported from
rag-math-test/rank-embedding-math/running-rerankers/sabermath_qwen3_reranker.py.
"""

from typing import ClassVar

from .base import ModelProcessor

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B"

# The paper's setting is a mathematical-relevance reranking task, so we use a
# task-specific instruction (the model card recommends customizing `instruct`
# per scenario; its default is the generic web-search instruction).
DEFAULT_TASK_INSTRUCTION = (
    "Given a math problem query, retrieve documents that are mathematically "
    "relevant to the query"
)


class Qwen3RerankerProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "qwen3-reranker"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
        batch_size: int = 8,
        max_length: int = 8192,
        use_flash_attn: bool = False,
    ) -> None:
        self._model_name = model_name
        self._task_instruction = task_instruction
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_flash_attn = use_flash_attn

        self._tokenizer = None
        self._model = None
        self._token_true_id = None
        self._token_false_id = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, padding_side="left"
        )

        if self._use_flash_attn:
            # "We recommend enabling flash_attention_2 for better
            #  acceleration and memory saving." (model card)
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                )
                .cuda()
                .eval()
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()

        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. Note "
            'that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self._task_instruction}\n"
            f"<Query>: {query}\n<Document>: {doc}"
        )

    def _process_inputs(self, pairs: list[str]):
        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(
                self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
            ),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self._prefix_tokens + ele + self._suffix_tokens
        inputs = self._tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self._max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        return inputs

    def _compute_scores(self, inputs) -> list[float]:
        import torch

        with torch.no_grad():
            batch_scores = self._model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()

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

        pairs = [self._format_instruction(query, doc) for doc in documents]

        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            inputs = self._process_inputs(batch)
            scores.extend(self._compute_scores(inputs))

        return scores
