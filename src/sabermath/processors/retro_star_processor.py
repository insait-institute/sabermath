"""Retro* (ljw13/retro-star-qwen3-{8b,32b}-0928): a GENERATIVE POINTWISE
reranker served via vLLM.

For each (query, document) pair the model is asked - in one chat turn - to
analyze the query, analyze the document, and conclude with an integer
relevance score in 0-100 wrapped in <score></score> tags. The parsed integer
IS the relevance score; there is no logprob head (contrast Rank1Processor,
which reads P(true) off the final-token logprobs).

Every constant below is transcribed from the team's own released evaluation
code, not from the model card's abbreviated snippet - fetched and read on
2026-08-28 from
https://github.com/VectorSpaceLab/agentic-search, files
`Retro-star/evaluation/bright/reranker_sglang.py` (SGLangReasoningLLMReranker:
prompt template, sampling params, truncation, score parsing/aggregation) and
`Retro-star/evaluation/bright/reranker_prompts.py`
(ReasoningBrightShortInstructions: the per-dataset query_type/doc_type/
relevance_definition triples). PROMPT_TEMPLATE is byte-identical to
`get_prompt_template()` there.

Where this differs from the model card: the card's snippet uses
max_new_tokens=1024, but the released BRIGHT scripts
(`scripts/.../100@1-30@16-retro-star-qwen3-8b-0928.sh`) pass
--reranker_max_new_tokens 4096, so 4096 is the protocol default here.

RELEVANCE DEFINITION. This prompt has no "no instruction" mode - a relevance
definition is a required slot, and the vendor ships one per BRIGHT dataset.
We use their `aops` triple verbatim (query = "math problem", document =
"math problem solution"), which is the exact shape of this benchmark's
statement-full task: a problem statement scored against problem+solution
documents. Note the mismatch on the OTHER two tasks - statement-statement
documents carry no solution, and full-full queries carry one - the doc_type
string is simply less accurate there; it is left alone rather than varied
per task, so one model's rows stay mutually comparable and match a real
vendor configuration.

DEVIATIONS, both deliberate:
  - BACKEND: vLLM, not SGLang (this repo has one serving stack, pinned in
    scripts/rerankers/envs/env_vllm.yml). The generation contract is
    reproduced knob for knob: n/temperature/top_k/repetition_penalty/
    max_new_tokens/skip_special_tokens/spaces_between_special_tokens, the
    same chat template with enable_thinking=False, and the same
    context_length - max_new_tokens - 128 prompt budget.
  - SEEDING: the official code seeds the ENGINE once (sglang
    random_seed=42), which makes a run reproducible only if batching is
    also identical - it is not, across a checkpoint/resume. We seed each
    REQUEST from a hash of its own (query, document) text instead, so a
    given pair scores identically regardless of batch composition, query
    order, or where a resumed run picks up. Same policy, same reason, as
    INFXRetrieverProcessor's per-query rewrite seeding.
"""

import hashlib
import re
from typing import ClassVar

from .base import ModelProcessor

DEFAULT_MODEL = "ljw13/retro-star-qwen3-32b-0928"

# reranker_sglang.py defaults: context_length=32768, and
# max_prompt_length = context_length - max_new_tokens - 128.
DEFAULT_CONTEXT_LENGTH = 32768
# The released BRIGHT scripts' --reranker_max_new_tokens.
DEFAULT_MAX_NEW_TOKENS = 4096
# --reranker_sample_k in the released scripts' first ("100@1") pass. Their
# second pass re-scores only the top 30 with sample_k=16 and averages; this
# harness scores every candidate exactly once per run, so the faithful
# single-pass equivalent is sample_k=1. Raising it here averages k samples
# per pair, exactly as their `score / self.sample_k` does.
DEFAULT_SAMPLE_K = 1

SCORE_PATTERN = re.compile(r"<score>\s*(\d+)\s*</score>")

# Byte-identical to SGLangReasoningLLMReranker.get_prompt_template().
PROMPT_TEMPLATE = """\
Here is the **relevance definition** in a retrieval task: {relevance_definition}

Now given a **query** ({query_type}) and a **document** ({doc_type}) in this retrieval task, your mission is to perform the following steps.

1. Query Analysis: Think to reason and describe what information would be most helpful in answering the query.
2. Document Analysis: Discuss how the information provided by the document fulfills or fails to fulfill the requirements implied by the query.
3. Relevance Annotation: Based on the relevance definition and the insights from the previous two steps, clearly justify your final relevance annotation result and annotate an integer score from a scale of 0 to 100. Please use the following guide:
    - **80-100 (Highly Relevant):** The document directly and comprehensively addresses the query's intent. It is a core and authoritative answer.
    - **60-80 (Relevant):** The document substantially addresses the query's intent, providing most of the key information, but might miss some minor details.
    - **40-60 (Moderately Relevant):** The document is on-topic and addresses a part of the query's intent, but it is not a comprehensive answer.
    - **20-40 (Slightly Relevant):** The document mentions keywords from the query, but its main topic is different. It offers very limited value.
    - **0-20 (Irrelevant):** The document does not address the query's intent at all and is off-topic.

After providing your detailed analysis and justification for all the steps above, conclude your entire response with the final relevance score. The score must be placed strictly between the <score> tags. There should be no other text or explanation inside the tags:
<score>
[From a scale of 0 to 100, annotate the degree of relevance between the query and the document.]
</score>

Query ({query_type}):
[Begin of Query]
{query}
[End of Query]

Document ({doc_type}):
[Begin of Document]
{doc}
[End of Document]
"""

# reranker_prompts.py -> ReasoningBrightShortInstructions["aops"], verbatim.
SABERMATH_QUERY_TYPE = "math problem"
SABERMATH_DOC_TYPE = "math problem solution"
SABERMATH_RELEVANCE_DEFINITION = (
    "Given a query (math problem) and a document (math problem solution), "
    "the document is relevant to the query if the theorems used in the "
    "document can provide helpful insights for solving the problem in the "
    "query."
)


class RetroStarProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "retro-star"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tensor_parallel_size: int = 1,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        sample_k: int = DEFAULT_SAMPLE_K,
        gpu_memory_utilization: float = 0.9,
        disable_custom_all_reduce: bool = True,
        enable_thinking: bool = False,
        query_type: str = SABERMATH_QUERY_TYPE,
        doc_type: str = SABERMATH_DOC_TYPE,
        relevance_definition: str = SABERMATH_RELEVANCE_DEFINITION,
    ) -> None:
        self._model_name = model_name
        self._tensor_parallel_size = tensor_parallel_size
        self._context_length = context_length
        self._max_new_tokens = max_new_tokens
        self._sample_k = sample_k
        self._gpu_memory_utilization = gpu_memory_utilization
        self._disable_custom_all_reduce = disable_custom_all_reduce
        self._enable_thinking = enable_thinking
        self._query_type = query_type
        self._doc_type = doc_type
        self._relevance_definition = relevance_definition

        # SGLangReasoningLLMReranker.__init__, verbatim.
        self._max_prompt_length = context_length - max_new_tokens - 128

        self._llm = None
        self._tokenizer = None

    @property
    def model(self) -> str | None:
        return self._model_name

    def _init(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("Please install vllm to use RetroStarProcessor") from e
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # No dtype pin: the checkpoint declares bfloat16 and vLLM's "auto"
        # keeps it. (rank1's float16 pin is inherited from ITS model card;
        # nothing in Retro*'s asks for a downcast.)
        #
        # disable_custom_all_reduce defaults ON here because vLLM's custom
        # all-reduce kernel does NOT work on this cluster at tp=4. Observed
        # (job 761561, msp3-2, 32B at tp=4): all four workers print
        #   Failed: Cuda error .../custom_all_reduce.cuh:455 'invalid argument'
        # during kernel warmup, a worker then dies with exit code None (a
        # signal, no Python exception), and engine init fails with the
        # misleading "Engine core initialization failed" / shm_broadcast
        # "cancelled" further up the stack. That kernel needs working
        # peer-to-peer access between every pair of GPUs in the group; the
        # tp=2 8B run on msp3-4 is unaffected, so this is specific to the
        # wider group. Falling back to NCCL costs a little all-reduce latency
        # and nothing else. Pass False only on a node where P2P is known good.
        self._llm = LLM(
            model=self._model_name,
            tensor_parallel_size=self._tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self._context_length,
            gpu_memory_utilization=self._gpu_memory_utilization,
            disable_custom_all_reduce=self._disable_custom_all_reduce,
        )

    def _sampling_params(self, seed: int):
        from vllm import SamplingParams

        # compute_score()'s sampling_params dict, plus the per-request seed
        # (see the module docstring's SEEDING note).
        return SamplingParams(
            n=self._sample_k,
            temperature=0.6,
            top_k=40,
            repetition_penalty=1.0,
            max_tokens=self._max_new_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            seed=seed,
        )

    def _truncate(self, text: str) -> str:
        """_truncate_texts() with max_length=max_prompt_length, the official
        default when no per-side truncation limit is configured. Never fires
        on this benchmark (the longest document is ~6k tokens against a
        28544-token budget), but kept so the two paths cannot diverge on a
        longer corpus."""
        ids = self._tokenizer(text)["input_ids"]
        if len(ids) <= self._max_prompt_length:
            return text
        return self._tokenizer.decode(ids[: self._max_prompt_length])

    def _build_input(self, query: str, document: str) -> str:
        prompt = PROMPT_TEMPLATE.format(
            relevance_definition=self._relevance_definition,
            query=self._truncate(query),
            doc=self._truncate(document),
            query_type=self._query_type,
            doc_type=self._doc_type,
        )
        # Second, whole-prompt truncation - also official, also inert here.
        ids = self._tokenizer(prompt)["input_ids"]
        if len(ids) > self._max_prompt_length:
            prompt = self._tokenizer.decode(ids[: self._max_prompt_length])
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )

    @staticmethod
    def _seed_for(query: str, document: str) -> int:
        digest = hashlib.sha256(f"{query}\x00{document}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    def _parse_score(self, text: str) -> int:
        """`int(re.search(...).group(1))`, 0 on any failure - the official
        `except: score += 0`. A miss means the model never emitted a
        parseable <score> tag (usually: it ran out of max_new_tokens
        mid-analysis), and the vendor treats that as minimum relevance."""
        match = SCORE_PATTERN.search(text)
        if match is None:
            return 0
        try:
            return int(match.group(1))
        except ValueError:
            return 0

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

        inputs = [self._build_input(query, doc) for doc in documents]
        params = [self._seed_for(query, doc) for doc in documents]

        # Production default (batch_size=None): one generate call over all of
        # this query's candidates, vLLM schedules internally - same shape as
        # Rank1Processor. batch_size exists for the timing harness.
        step = batch_size or len(inputs)
        outputs = []
        for start in range(0, len(inputs), step):
            chunk = inputs[start : start + step]
            outputs.extend(
                self._llm.generate(
                    chunk,
                    [self._sampling_params(s) for s in params[start : start + step]],
                    use_tqdm=show_progress_bar,
                )
            )

        # Mean over the k samples of one pair, matching `score / sample_k`.
        return [
            sum(self._parse_score(o.text) for o in out.outputs) / len(out.outputs)
            for out in outputs
        ]


class RetroStarRewrittenProcessor(ModelProcessor):
    """Retro* scoring the REWRITTEN query instead of the raw problem
    statement - Reason-Rewriter feeding the reranker rather than the
    retriever.

    THIS IS OUR OWN CONSTRUCTION, NOT THE VENDOR'S PIPELINE, and the
    distinction is easy to get wrong. In BGE-Reasoner the rewrite drives
    RETRIEVAL only: their rerank stage's data loader
    (Retro-star/evaluation/bright/data_loader.py) reads the ORIGINAL queries
    from `xlangai/bright`, and `--retrieval_split
    reasoner-rewritten-query-0821` merely selects which search results to
    rerank. Reproducing that faithfully on this benchmark would be a no-op:
    every query here has a FIXED 150-document candidate set, so there is no
    retrieval stage for a rewrite to influence, and Retro* would see exactly
    the same query and the same candidates as the plain retro-star-* row.
    This entry therefore asks a DIFFERENT question - does a long
    reasoning-style query help a generative reranker score pairs? - and its
    number must not be reported as the vendor's cascade.

    The rewrites are the very same ones the reason-rewriter-reason-embed-8b
    row used: same processor, same recipe fingerprint, same log file. With
    that log present every query is a cache hit, so the 7B generator is
    never loaded and this run costs no more than a plain Retro* run.

    Which of the five samples: index 0. The five exist so their EMBEDDINGS
    can be averaged (see ReasonRewriterProcessor fact 2); a generative
    reranker consumes one text and cannot average, and scoring all five
    would multiply an already per-pair generation cost by five. Sample 0 is
    deterministic given the per-query seed.
    """

    processor: ClassVar[str | None] = "retro-star-rewritten"

    def __init__(
        self,
        model_name: str = "ljw13/retro-star-qwen3-8b-0928",
        *,
        tensor_parallel_size: int = 1,
        rewrite_log_path: str | None = None,
        rewrite_sample: int = 0,
        gpu_memory_utilization: float = 0.55,
        rewriter_gpu_memory_utilization: float = 0.30,
        require_cached_rewrites: bool = False,
        **retro_kwargs,
    ) -> None:
        from .reason_rewriter_processor import ReasonRewriterProcessor

        self._rewrite_sample = rewrite_sample
        self._require_cached = require_cached_rewrites
        # Memory is split for the case where the log does NOT already cover a
        # query and the 7B generator has to load beside the reranker. When
        # every query is cached (the normal case) the generator never loads
        # and the reranker simply uses less than it could.
        self._rewriter = ReasonRewriterProcessor(
            rewrite_log_path=rewrite_log_path,
            rewriter_gpu_memory_utilization=rewriter_gpu_memory_utilization,
        )
        self._retro = RetroStarProcessor(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **retro_kwargs,
        )

    @property
    def model(self) -> str:
        return f"{self._rewriter._rewriter_name} + {self._retro.model}"

    def prefetch_rewrites(self, queries: list[str]) -> None:
        """Harness hook (see run_rerankers._prefetch_if_supported). With the
        rewriter row's log in place this reports everything cached and
        generates nothing.

        require_cached_rewrites turns a missing rewrite into an immediate,
        legible error instead of a confusing CUDA OOM. It exists for the 32B
        variant, which claims most of the GPU for the reranker precisely
        BECAUSE the generator is never expected to load; if a query were
        missing, that generator would try to load into the sliver that is
        left and die with an allocator message that says nothing about
        rewrites."""
        if self._require_cached:
            missing = self._rewriter._missing(list(queries))
            if missing:
                raise RuntimeError(
                    f"{len(missing)} of {len(queries)} queries have no cached "
                    "rewrite, and this entry runs with the reranker holding "
                    "most of the GPU, so the rewriter cannot be loaded to "
                    "generate them. Run the reason-rewriter-reason-embed-8b "
                    "row first (it writes the shared rewrite log), or "
                    "construct this processor with "
                    "require_cached_rewrites=False and a memory split that "
                    "fits both models."
                )
        self._rewriter.prefetch_rewrites(queries)

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
        rewrites = self._rewriter._rewrite(
            query,
            check_cache=kwargs.get("check_cache", True),
            update_cache=kwargs.get("update_cache", True),
        )
        rewritten = rewrites[self._rewrite_sample]
        return self._retro.get_scores(
            rewritten,
            documents,
            show_progress_bar=show_progress_bar,
            **{k: v for k, v in kwargs.items() if k not in ("check_cache", "update_cache")},
        )
