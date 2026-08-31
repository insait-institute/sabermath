"""Qwen/Qwen3-Reranker-* served through vLLM - the PRODUCTION path.

Uses the model card's official vLLM recipe: the causal checkpoint is reused
as a binary ("no"/"yes") sequence classifier via hf_overrides
(Qwen3ForSequenceClassification + classifier_from_token +
is_original_qwen3_reranker), and each (query, document) pair is scored with
llm.score() on the <Instruct>/<Query>/<Document> template below.

Validated against an HF-transformers implementation of the same template on
2026-08-20 - FEASIBLE for all three sizes (Spearman >= 0.999, mean
|dNDCG@10| <= 0.026 on the frozen comparison sample). That reference was
removed on 2026-08-31; the verdicts are archived in
results/diagnostics/vllm_feasibility/summary.json, and the template strings
it shared now live here, which is the only remaining copy.
"""

from typing import ClassVar

from .base import ModelProcessor

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B"

# A mathematical-relevance reranking task, so this is a task-specific
# instruction (the model card recommends customizing `instruct` per scenario;
# its own default is the generic web-search one, which the registry uses as
# the p0 baseline instead - see VENDOR_QWEN3_RERANKER_INSTRUCTION).
DEFAULT_TASK_INSTRUCTION = (
    "Given a math problem query, retrieve documents that are mathematically "
    "relevant to the query"
)

# The official yes/no judging chat template, verbatim from the model card
# (https://huggingface.co/Qwen/Qwen3-Reranker-8B).
PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the "
    "requirements based on the Query and the Instruct provided. Note "
    'that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_MAX_LENGTH = 8192  # the HF reference implementation's default


class Qwen3RerankerVLLMProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "qwen3-reranker-vllm"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        # batch_size: client-side slicing of llm.score calls. None (the
        # production default) scores all candidates in one call and lets
        # vLLM schedule; the timing harness passes 16 to standardize
        # request-level parallelism.
        self._model_name = model_name
        self._task_instruction = task_instruction
        self._max_length = max_length
        self._batch_size = batch_size
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization

        self._llm = None
        self._tokenizer = None
        self._suffix_len = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "Please install vllm to use Qwen3RerankerVLLMProcessor"
            ) from e
        from transformers import AutoTokenizer

        hf_overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        }
        base = dict(
            model=self._model_name,
            hf_overrides=hf_overrides,
            max_model_len=self._max_length,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
        )
        # vllm==0.26.0 spells the pooling runner `runner="pooling"`; older
        # builds used `task="score"` - tolerate both rather than hard-pinning
        # the harness code to one spelling.
        try:
            self._llm = LLM(runner="pooling", **base)
        except TypeError:
            self._llm = LLM(task="score", **base)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._suffix_len = len(
            self._tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
        )

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        batch_size: int | None = None,
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []

        self._init()

        query_part = (
            PROMPT_PREFIX
            + f"<Instruct>: {self._task_instruction}\n<Query>: {query}\n"
        )
        # The HF processor truncates each pair with longest_first, which in
        # practice trims the (long) document; mirror that by capping the
        # document to whatever budget the query part leaves (small margin for
        # the "<Document>: " tokens). Necessary client-side because vLLM
        # hard-rejects overlong prompts instead of truncating.
        query_len = len(self._tokenizer.encode(query_part, add_special_tokens=False))
        budget = self._max_length - query_len - self._suffix_len - 8

        doc_parts = []
        for doc in documents:
            ids = self._tokenizer.encode(doc, add_special_tokens=False)
            if len(ids) > budget:
                doc = self._tokenizer.decode(
                    ids[:budget],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            doc_parts.append(f"<Document>: {doc}" + PROMPT_SUFFIX)

        effective_batch = batch_size or self._batch_size or len(doc_parts)
        scores: list[float] = []
        for start in range(0, len(doc_parts), effective_batch):
            outputs = self._llm.score(
                query_part,
                doc_parts[start : start + effective_batch],
                use_tqdm=show_progress_bar,
            )
            scores.extend(float(o.outputs.score) for o in outputs)
        return scores
