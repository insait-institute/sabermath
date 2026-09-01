import hashlib
from typing import ClassVar

import numpy as np

from .base import ModelProcessor
from .vllm_processor import VLLMProcessor

REWRITER_REPO = "cfli/reasoner-rewriter-qwen2.5-7b-0821"
RETRIEVER_REPO = "hanhainebula/reason-embed-qwen3-8b-0928"

# Model card, verbatim (note the literal backticks around the tag names).
PROMPT_TEMPLATE = """\
Given a task and an input, first analyze the task and the input within the `<think>` and `</think>` tags. In your analysis:
- Break down the requirements of the task
- Identify key components from the input
- Think step by step to reason about what should be included in the output

Then, within the `<response>` and `</response>` tags, present the complete long output.

## Task
{task}

## Input
{query}"""

# task_desc["aops"] from the model card's per-dataset map - the math-problem
# task description, i.e. the one whose queries look like this benchmark's.
SABERMATH_TASK = (
    "Generate the functions and techniques involved in solving this math "
    "problem, and provide a detailed explanation of the functions and "
    "techniques."
)

# The model card's explicit knobs, plus the ones that fall through from its
# generation_config.json.
REWRITER_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.05,
    # 8192, not the card's 4096: lowering it truncates the rewrite mid-reasoning,
    # which is invisible in the scores.
    "max_tokens": 8192,
}
# The released artifact ships five samples per query.
DEFAULT_N_REWRITES = 5

# The retriever's own GENERIC_MODELS configuration, duplicated rather than
# imported so this processor stays importable on its own. Keep the two in sync.
RETRIEVER_POOLER_CONFIG = {"pooling_type": "LAST", "normalize": True}


class ReasonRewriterProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "reason-rewriter"

    # Bump when the rewrite recipe changes (prompt, task text, sampling,
    # sample count, seeding); that invalidates persisted rewrite logs.
    #
    # THE BUMP IS NOT ENOUGH ON ITS OWN. The per-query nDCG checkpoint resumes
    # BEFORE a processor is asked for any score, and it is keyed by (model,
    # subset, task) with no knowledge of a recipe - so on an already-scored
    # subset the fingerprint is never consulted and the change is a silent
    # no-op. Delete this model's checkpoint directory too.
    _REWRITE_RECIPE_FINGERPRINT = "v2:sha256-seed:n5:t0.6:p0.9:k20:rp1.05:8192"

    def __init__(
        self,
        rewriter_name: str = REWRITER_REPO,
        retriever_name: str = RETRIEVER_REPO,
        *,
        n_rewrites: int = DEFAULT_N_REWRITES,
        task: str = SABERMATH_TASK,
        rewrite_log_path: str | None = None,
        tensor_parallel_size: int = 1,
        # Split evenly: with the batched prefetch the rewriter is the
        # bottleneck and wants KV cache, while the retriever only runs short
        # pooling passes. The 0.90 total matches one default vLLM engine.
        rewriter_gpu_memory_utilization: float = 0.45,
        retriever_gpu_memory_utilization: float = 0.45,
        # Retriever half only. The pooler is deliberately not overridable
        # (every Reason-Embed checkpoint is LAST+normalize); max_model_len is,
        # because a composed row taking llama's 131072 default would not be
        # comparable to the standalone spec's 40960.
        retriever_init_kwargs: dict | None = None,
    ) -> None:
        self._rewriter_name = rewriter_name
        self._retriever_name = retriever_name
        self._n_rewrites = n_rewrites
        self._task = task
        self._rewrite_log_path = rewrite_log_path
        self._tensor_parallel_size = tensor_parallel_size
        self._rewriter_gpu_memory_utilization = rewriter_gpu_memory_utilization
        self._retriever_gpu_memory_utilization = retriever_gpu_memory_utilization
        self._retriever_init_kwargs = dict(retriever_init_kwargs or {})

        self._rewriter = None
        self._rewriter_tokenizer = None
        self._retriever: VLLMProcessor | None = None

        self._rewrite_cache: dict[str, list[str]] = self._load_rewrite_log()
        self._unsaved = 0

    @property
    def model(self) -> str:
        return f"{self._rewriter_name} + {self._retriever_name}"


    def _init_rewriter(self) -> None:
        if self._rewriter is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "Please install vllm to use ReasonRewriterProcessor"
            ) from e
        from transformers import AutoTokenizer

        self._rewriter_tokenizer = AutoTokenizer.from_pretrained(self._rewriter_name)
        self._rewriter = LLM(
            model=self._rewriter_name,
            tensor_parallel_size=self._tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=self._rewriter_gpu_memory_utilization,
        )

    def _init_retriever(self) -> None:
        if self._retriever is not None:
            return
        self._retriever = VLLMProcessor.from_huggingface(
            self._retriever_name,
            pooler_config=dict(RETRIEVER_POOLER_CONFIG),
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._retriever_gpu_memory_utilization,
            **self._retriever_init_kwargs,
        )

    def _init(self) -> None:
        self._init_rewriter()
        self._init_retriever()


    def _load_rewrite_log(self) -> dict[str, list[str]]:
        import json
        from pathlib import Path

        if not self._rewrite_log_path:
            return {}
        p = Path(self._rewrite_log_path)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        if (
            data.get("fingerprint") != self._REWRITE_RECIPE_FINGERPRINT
            or data.get("rewriter") != self._rewriter_name
            or data.get("task") != self._task
        ):
            # Stale recipe - regenerating is always safe (and correct).
            return {}
        rewrites = data.get("rewrites", {})
        if not isinstance(rewrites, dict):
            return {}
        return {k: list(v) for k, v in rewrites.items() if isinstance(v, list)}

    # Rewriting the whole log after every query is quadratic in bytes written
    # (~60GB over a 1000-query run), so batch the writes. Per-query seeding
    # makes anything lost on a crash regenerate byte-identically.
    _SAVE_EVERY = 25

    def _maybe_save_rewrite_log(self, *, force: bool = False) -> None:
        if not self._rewrite_log_path or self._unsaved == 0:
            return
        if force or self._unsaved >= self._SAVE_EVERY:
            self._save_rewrite_log()
            self._unsaved = 0

    def _save_rewrite_log(self) -> None:
        import json
        from pathlib import Path

        if not self._rewrite_log_path:
            return
        p = Path(self._rewrite_log_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "fingerprint": self._REWRITE_RECIPE_FINGERPRINT,
                        "rewriter": self._rewriter_name,
                        "task": self._task,
                        "rewrites": self._rewrite_cache,
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            tmp.replace(p)
        except OSError as e:
            # Auditability must never kill a scoring run.
            print(f"[!!] reason-rewriter: rewrite log write failed ({e})")


    def _generate(self, queries: list[str], *, show_progress: bool) -> None:
        from vllm import SamplingParams

        self._init_rewriter()

        prompts, params = [], []
        for q in queries:
            prompts.append(
                self._rewriter_tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(
                                task=self._task, query=q
                            ),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            seed = int.from_bytes(
                hashlib.sha256(q.encode("utf-8")).digest()[:4], "big"
            )
            params.append(
                SamplingParams(n=self._n_rewrites, seed=seed, **REWRITER_SAMPLING)
            )

        outputs = self._rewriter.generate(prompts, params, use_tqdm=show_progress)
        for q, out in zip(queries, outputs):
            # Guard OUR pipeline: an empty rewrite would zero-norm-crash the
            # cosine downstream. Fall back per-sample to the original query
            # rather than dying partway into a run.
            self._rewrite_cache[q] = [
                (o.text if o.text.strip() else q) for o in out.outputs
            ]
            self._unsaved += 1

    def _missing(self, queries: list[str]) -> list[str]:
        out, seen = [], set()
        for q in queries:
            if q in seen:
                continue
            seen.add(q)
            cached = self._rewrite_cache.get(q)
            if not cached or len(cached) != self._n_rewrites:
                out.append(q)
        return out

    def prefetch_rewrites(self, queries: list[str]) -> None:
        missing = self._missing(queries)
        if not missing:
            print(
                f"[~] reason-rewriter: all {len(queries)} queries already "
                "have cached rewrites - nothing to prefetch."
            )
            return
        print(
            f"[~] reason-rewriter: prefetching rewrites for {len(missing)} "
            f"queries x {self._n_rewrites} samples in one batched call "
            f"({len(queries) - len(missing)} already cached)..."
        )
        self._generate(missing, show_progress=True)
        self._maybe_save_rewrite_log(force=True)

    def _rewrite(
        self, query: str, *, check_cache: bool = True, update_cache: bool = True
    ) -> list[str]:
        if check_cache:
            cached = self._rewrite_cache.get(query)
            if cached and len(cached) == self._n_rewrites:
                return cached

        self._generate([query], show_progress=False)
        rewrites = self._rewrite_cache[query]

        if update_cache:
            self._maybe_save_rewrite_log()
        else:
            self._rewrite_cache.pop(query, None)
            self._unsaved = max(0, self._unsaved - 1)
        return rewrites


    # WHICH HALF GETS THE INSTRUCTION. A prompt is wrapped into the query text,
    # which is what _rewrite() consumes, so the REWRITER is instructed for free
    # while the encoder never sees it - the query side embeds the rewrites, not
    # the query. Reapplying the wrap here is the only way to instruct the
    # encoder half too. Left None, this is the prompt-free encoder.
    @staticmethod
    def _texts_to_embed(
        rewrites: list[str],
        embed_instruction: str | None,
        embed_instruction_template: str,
    ) -> list[str]:
        if embed_instruction is None:
            return rewrites
        from sabermath.instructions import format_instructed_query

        return [
            format_instructed_query(
                embed_instruction, rewrite, embed_instruction_template
            )
            for rewrite in rewrites
        ]


    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        check_cache: bool = True,
        update_cache: bool = True,
        embed_instruction: str | None = None,
        embed_instruction_template: str = "canonical",
        **kwargs,
    ) -> list[float]:
        if not documents:
            return []

        self._init()

        rewrites = self._rewrite(
            query, check_cache=check_cache, update_cache=update_cache
        )

        # Encode every rewrite, then mean-pool. Deliberately uncached: the
        # expensive half is, and these vectors are used once per query.
        query_embeddings = np.asarray(
            self._retriever.encode(
                self._texts_to_embed(
                    rewrites, embed_instruction, embed_instruction_template
                ),
                show_progress_bar=False,
                **kwargs,
            ),
            dtype=float,
        )
        query_embedding = query_embeddings.mean(axis=0)

        # Straight through the retriever's own vector cache, so a document
        # shared between queries or tasks is embedded once.
        if not hasattr(self._retriever, "_vector_cache"):
            self._retriever._vector_cache = {}
        document_embeddings = self._retriever._encode_side(
            documents,
            tag=None,
            show_progress_bar=show_progress_bar,
            check_cache=check_cache,
            update_cache=update_cache,
            encode_kwargs=kwargs,
        )

        # Cosine, not dot product: mean-pooling unit vectors does not return
        # a unit vector.
        return self._retriever._cosine_similarity(
            query_embedding, np.asarray(document_embeddings, dtype=float)
        )


    def export_cache(self, path: str) -> None:
        self._init()
        self._retriever.export_cache(path)

    def import_cache(self, path: str) -> None:
        self._init()
        self._retriever.import_cache(path)

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self._init_retriever()
        return self._retriever.encode(texts, **kwargs)

    def encode_queries(
        self,
        texts: list[str],
        *,
        embed_instruction: str | None = None,
        embed_instruction_template: str = "canonical",
        **kwargs,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)

        self.prefetch_rewrites(list(texts))
        self._init_retriever()

        # Flatten to one embedding call: [q0r0..q0rN, q1r0..q1rN, ...]
        flat, counts = [], []
        for t in texts:
            rewrites = self._texts_to_embed(
                self._rewrite(t), embed_instruction, embed_instruction_template
            )
            counts.append(len(rewrites))
            flat.extend(rewrites)
        embs = np.asarray(self._retriever.encode(flat, **kwargs), dtype=float)

        out = np.empty((len(texts), embs.shape[1]), dtype=float)
        at = 0
        for i, c in enumerate(counts):
            out[i] = embs[at : at + c].mean(axis=0)
            at += c
        return out
