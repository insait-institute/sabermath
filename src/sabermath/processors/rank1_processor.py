"""jhu-clsp/rank1-32b: a reasoning reranker served via vLLM.

For each (query, document) pair, the model generates a reasoning chain inside
a <think>...</think> section, then emits a binary relevance judgment ("true"
or "false"); the continuous relevance score is P(true) / (P(true) + P(false))
computed from the final-token logprobs. This follows the officially supplied
minimal vLLM example from the Hugging Face model card
(https://huggingface.co/jhu-clsp/rank1-32b): same prompt template,
SamplingParams (temperature=0, max_tokens=8192, logprobs=20,
stop=["</think> true", "</think> false"]), and true/false logprob scoring.

Ported from rag-math-test/rank-embedding-math/running-rerankers/sabermath_rank1.py
(verified against a completed 100-query run: Mean nDCG@10 = 0.6068). That
standalone script's larger runs (400+ and 1000 queries) were repeatedly
killed by SLURM wall-clock limits before finishing - the per-query cost of
`max_thinking_tokens` x ~150 candidates is high. Tune `max_thinking_tokens`,
`tensor_parallel_size` and/or evaluate on a query subset (see
scripts/run_rerankers.py --n) accordingly for a full run.
"""

import math
from typing import ClassVar

from .base import ModelProcessor

DEFAULT_MODEL = "jhu-clsp/rank1-32b"
DEFAULT_MAX_MODEL_LEN = 16000


class Rank1Processor(ModelProcessor):
    processor: ClassVar[str | None] = "rank1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "float16",
        max_thinking_tokens: int = 8192,
    ) -> None:
        self._model_name = model_name
        self._max_model_len = max_model_len
        self._max_thinking_tokens = max_thinking_tokens
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype

        self._llm = None
        self._sampling_params = None
        self._tokenizer = None
        self._true_token = None
        self._false_token = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "Please install vllm to use Rank1Processor"
            ) from e
        from transformers import AutoTokenizer

        # NOTE: tensor_parallel_size > 1 and enable_prefix_caching=False were
        # both tried here at various points (the latter to address a
        # per-query slowdown - vllm-project/vllm#16985, #28726, #3154) and
        # BOTH have since been confirmed, via a direct diagnostic re-run of
        # the exact original single-GPU/default-caching config on known
        # query indices, to silently corrupt this model's relevance scores
        # (nDCG collapsing from ~0.61 to ~0.41 in production; individual
        # queries dropping from ~0.85 to ~0.15-0.33). Root cause not fully
        # isolated between the two (may be one or both), but since neither
        # is needed for correctness, both are left at vLLM's defaults
        # (tensor_parallel_size=1, enable_prefix_caching=True) rather than
        # re-attempting either optimization. Do not reintroduce either
        # without re-validating against known-good per-query scores first.
        self._llm = LLM(
            model=self._model_name,
            tensor_parallel_size=self._tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            dtype=self._dtype,
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self._max_thinking_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._true_token = self._tokenizer(
            " true", add_special_tokens=False
        ).input_ids[0]
        self._false_token = self._tokenizer(
            " false", add_special_tokens=False
        ).input_ids[0]

    @staticmethod
    def _create_prompt(query: str, document: str) -> str:
        return (
            "Determine if the following passage is relevant to the query. "
            "Answer only with 'true' or 'false'.\n"
            f"Query: {query}\n"
            f"Passage: {document}\n"
            "<think>"
        )

    def _truncate_document(self, query: str, document: str) -> str:
        """Truncate the document (by tokens) so the prompt fits the context
        window, reserving room for the generated reasoning chain."""
        reserve = self._sampling_params.max_tokens + 64
        overhead = len(
            self._tokenizer(
                self._create_prompt(query, ""), add_special_tokens=False
            ).input_ids
        )
        budget = self._max_model_len - reserve - overhead
        doc_ids = self._tokenizer(document, add_special_tokens=False).input_ids
        if budget > 0 and len(doc_ids) > budget:
            document = self._tokenizer.decode(doc_ids[:budget])
        return document

    def _relevance_score(self, output) -> float:
        """Extract P(true) from a vLLM completion, per the official example."""
        final_logits = output.logprobs[-1]

        true_lp = final_logits.get(self._true_token)
        false_lp = final_logits.get(self._false_token)

        # Edge-case fallback: if a token is missing from the top-20 logprobs,
        # treat it as ~impossible (the official repo handles this more
        # elaborately).
        true_logit = true_lp.logprob if true_lp is not None else -1e4
        false_logit = false_lp.logprob if false_lp is not None else -1e4

        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        denom = true_score + false_score
        return 0.5 if denom == 0 else true_score / denom

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

        prompts = [
            self._create_prompt(query, self._truncate_document(query, doc))
            for doc in documents
        ]

        # All candidates of this query are scored in a single batched call.
        outputs = self._llm.generate(
            prompts, self._sampling_params, use_tqdm=show_progress_bar
        )

        return [self._relevance_score(out.outputs[0]) for out in outputs]
