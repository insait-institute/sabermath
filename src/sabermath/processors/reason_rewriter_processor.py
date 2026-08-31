"""Reason-Rewriter (cfli/reasoner-rewriter-qwen2.5-7b-0821) composed with a
Reason-Embed retriever - the BGE-Reasoner "rewrite, then embed" pipeline as a
single benchmark entry.

The rewriter turns each query into a long reasoning-and-answer text; that
text (not the original query) is what gets embedded and scored against the
untouched documents. Scoring is delegated to the SAME vLLM embedding
processor the standalone reason-embed-qwen3-8b row uses, with the SAME
prompt-free protocol, so a standalone-vs-composed comparison isolates the
rewrite and nothing else.

Sources, all read on 2026-08-28:
  - the model card (https://huggingface.co/cfli/reasoner-rewriter-qwen2.5-7b-0821):
    PROMPT_TEMPLATE, the per-BRIGHT-dataset `task_desc` map, and the
    generation call;
  - the checkpoint's generation_config.json (top_k=20,
    repetition_penalty=1.05, and the temperature/top_p defaults the card
    overrides);
  - the released rewrite artifact
    (https://huggingface.co/datasets/cfli/reasoner-rewritten-query-0821),
    which is what their downstream retrieval actually consumed;
  - https://github.com/VectorSpaceLab/agentic-search,
    `ReasonRewriter/evaluation/FlagEmbedding-offical/{data_loader,searcher}.py`,
    which load that artifact and consume it.

Four facts about the official pipeline are worth stating explicitly,
because each looks like a bug until you check the released artifact:

1. THE REWRITE IS THE WHOLE GENERATION, think block included. The card's
   snippet assigns `rewrite_query = tokenizer.batch_decode(...)` with no
   extraction, and every row of the released dataset's `query` field indeed
   starts with "<think>\\nOkay, ..." and ends with "</response>". The
   `<response>` body is NOT split out. data_loader.py then uses that field
   verbatim as the query text. So we embed the full text too - stripping to
   the response body would be a different (untested) system.

2. THERE ARE FIVE REWRITES PER QUERY, and their embeddings are MEAN-POOLED.
   The released dataset ships `query` as a list of 5 samples (8k-18k
   characters each), and searcher.py branches on that list, encodes every
   sample, and averages: `queries_emb.reshape(...).mean(axis=1)`. As
   published that line actually raises - it reshapes an (n_q*5, dim) array
   to (n_q, 5), dropping the embedding dimension entirely - but the intent
   is unambiguous, and their query ordering is sample-major, so the
   mean-over-samples is reproduced here directly instead of through their
   broken reshape.

3. THE SAMPLING KNOBS COME FROM TWO PLACES. The card's generate() call
   passes temperature=0.6 and top_p=0.9 explicitly, which override
   generation_config.json; top_k=20 and repetition_penalty=1.05 are NOT
   passed, so they fall through from that file. The effective recipe is the
   union, which is what REWRITER_SAMPLING encodes.

4. THE CARD'S max_new_tokens=4096 IS TOO SMALL, and this is the one card
   value deliberately NOT followed. MEASURED against the released artifact
   (200 aops rewrites = 40 queries x 5 samples, tokenized with this
   checkpoint's own tokenizer on 2026-08-28): all 200 close with
   "</response>", their length is median 4908 tokens, and 11% sit exactly at
   8194-8195 with nothing above - the signature of a generation budget of
   8192, not 4096. At 4096, 67.5% of the vendor's own rewrites would have
   been cut off. Confirmed empirically on our side too: a 5-query smoke run
   at 4096 (job 761495) produced five rewrites that ALL ran out of budget
   inside <think> and never emitted the <response> section at all - i.e. the
   embedded text was a truncated reasoning trace, missing the "complete long
   output" the prompt asks for. REWRITER_SAMPLING therefore uses 8192.

PERFORMANCE - WHY THERE IS A PREFETCH. Generating one query's rewrites at a
time puts exactly n_rewrites (5) sequences in flight, which leaves an H200
almost idle: MEASURED at ~0.6 min/query, i.e. ~10 hours for 1000 queries,
against ~13 queries/min for the Retro* rerankers, which submit all ~150 of a
query's candidates at once. prefetch_rewrites() therefore generates the
WHOLE query set in one batched call, letting vLLM's continuous batching keep
the GPU full; scoring then runs off the cache. scripts/run_experiments.py
calls it with the exact query set being evaluated (so --n subsets and query
shards prefetch only what they need), and a query that somehow misses the
cache still falls back to generating on demand, so the hook is an
optimization and never a correctness dependency.

Batching CANNOT change the rewrites: every request carries its own
content-derived seed (see SEEDING below), and vLLM seeds each request's
sampler independently of batch composition. A prefetched rewrite is
byte-identical to a lazily generated one, which is also why the recipe
fingerprint does not change for this.

This mirrors the vendor's own shape, incidentally: they precomputed every
rewrite offline and released it as a dataset rather than generating inline.

DEVIATIONS, all deliberate:
  - BACKEND: vLLM for the rewriter, not HF `model.generate`. 5 samples x
    ~1000 queries x up to 8192 new tokens is not a workload for
    one-sequence-at-a-time HF generation. Both models are resident in one
    process, so each gets a slice of the GPU (see the *_gpu_memory_utilization
    arguments) rather than vLLM's default 0.9.
  - SEEDING: the official pipeline seeds nothing, so its rewrites are
    irreproducible run to run. Each query's request is seeded from a hash of
    the query text, so a query always yields the same 5 rewrites regardless
    of order or where a resumed run picks up - identical policy and
    rationale to INFXRetrieverProcessor.
  - NO QUERY PROMPT on the embedder side. Reason-Embed ships an "Instruct:
    ...\\nQuery: " prompt and the vendor's own BRIGHT scripts use one, but
    this repo's standalone reason-embed-qwen3-8b row is prompt-free (the
    instruction dimension is studied separately, through --instructions), and
    the composed row must differ from it by the rewrite ALONE to be readable
    against it.

The task description slot is filled with the vendor's `aops` entry, the
math-problem member of their per-dataset `task_desc` map.
"""

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

# See fact (3) in the module docstring: card-explicit knobs plus the ones
# that fall through from generation_config.json.
REWRITER_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.05,
    # 8192, not the card's 4096 - see fact (4) in the module docstring. Raise
    # this only with evidence; lowering it silently truncates the rewrite
    # mid-reasoning, which is invisible in the scores.
    "max_tokens": 8192,
}
# See fact (2): the released artifact ships five samples per query.
DEFAULT_N_REWRITES = 5

# The retriever's own GENERIC_MODELS configuration in scripts/run_experiments.py -
# duplicated here rather than imported so this processor stays importable
# from the library alone; keep the two in sync.
RETRIEVER_POOLER_CONFIG = {"pooling_type": "LAST", "normalize": True}


class ReasonRewriterProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "reason-rewriter"

    # Bump when the rewrite recipe changes (prompt, task text, sampling,
    # sample count, seeding) - invalidates persisted rewrite logs.
    # v2 (2026-08-28): max_tokens 4096 -> 8192. The bump is what forces the
    # v1 logs - whose rewrites were truncated inside <think> - to be
    # discarded rather than warm-starting a new run with bad text.
    #
    # THIS FINGERPRINT IS NOT ENOUGH ON ITS OWN. Confirmed the hard way (job
    # 761576): after bumping it, the rerun reported the OLD score to the
    # digit, because evaluate()'s per-query nDCG checkpoint
    # (results/evaluation/.checkpoints/<model>/<subset>/<task>.json) resumes
    # BEFORE a processor is ever asked for a score, so no rewrite was
    # requested and the fingerprint was never consulted. That checkpoint is
    # keyed by (model, subset, task) only - it knows nothing about a recipe.
    # So when changing anything in REWRITER_SAMPLING / the prompt / the
    # sample count, delete this model's checkpoint directory too, or the
    # change is a silent no-op on any subset that has already been scored.
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
        # Split evenly rather than 0.30/0.55: with the batched prefetch the
        # rewriter is the throughput bottleneck and wants KV cache for as
        # many concurrent sequences as possible, while the retriever only
        # ever runs short pooling passes. 0.45+0.45 keeps the pair at the
        # same 0.90 total a single default vLLM engine would take.
        rewriter_gpu_memory_utilization: float = 0.45,
        retriever_gpu_memory_utilization: float = 0.45,
        # Extra vLLM init kwargs for the RETRIEVER half only. The pooler is
        # not overridable here on purpose (every Reason-Embed checkpoint is
        # LAST+normalize), but max_model_len is: the llama-3.1 variant is
        # pinned to 40960 in the registry's standalone spec so the ablation's
        # effective context matches across backends, and a composed row that
        # silently took llama's 131072 default would not be comparable to it.
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

    # ------------------------------------------------------------------ init

    def _init_rewriter(self) -> None:
        """Load the generator half ONLY. Split from the retriever so a caller
        that just wants rewrites - RetroStarRewrittenProcessor, or a run whose
        rewrites are all cached already - never pays for an 8B embedder it
        will not use."""
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
        """Load the embedding half ONLY."""
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
        """Both models live in one process, so they must SHARE the GPU: each
        vLLM engine is given an explicit fraction instead of the default 0.9,
        and the rewriter is built first so the retriever profiles against
        what is actually left. Lazy, so constructing this processor stays
        cheap and any load failure surfaces inside the run (where the
        harness's per-model subprocess isolation catches it)."""
        self._init_rewriter()
        self._init_retriever()

    # -------------------------------------------------------- rewrite log io

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

    # The log holds n_rewrites x ~12KB of text per query, so rewriting it
    # after every single query is quadratic in bytes written (~60GB over a
    # 1000-query run). Batch the writes; the log is an audit aid and a
    # warm-start, and per-query seeding makes anything lost regenerate
    # byte-identically.
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

    # ------------------------------------------------------------- rewriting

    def _generate(self, queries: list[str], *, show_progress: bool) -> None:
        """Generate and cache rewrites for every query given, in ONE vLLM
        call so continuous batching can keep the GPU busy. Per-request seeds
        make the result independent of how the queries are grouped."""
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
        """Queries with no complete cached rewrite set, deduplicated and in
        first-seen order."""
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
        """Rewrite this whole query set up front, in one batched call.

        Optional: scoring works without it (get_scores generates on demand),
        it is just ~an order of magnitude slower that way - see the
        PERFORMANCE note in the module docstring. Callers should pass the
        EXACT set being evaluated, so a subset run does not pay for queries
        it will never score."""
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
        """The query's n rewrites, generated once and cached by raw query
        text: one run_experiments.py process scores the same statement queries
        under two tasks, and both must see the same rewrites."""
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

    # ------------------------------------------------- instruction plumbing

    # WHICH HALF GETS THE INSTRUCTION. run_experiments.py's --instructions path
    # wraps the instruction INTO the query text ("Instruct: ...\nQuery: ..."),
    # and _rewrite() is handed that wrapped text - so the REWRITER is
    # instructed for free, and its cache key differs per prompt key. The
    # embedder never sees it: the query side embeds the rewrites, not the
    # query. Reapplying the wrap to each rewrite here is therefore the only
    # way to instruct the encoder half too, which is what the
    # reason-rewriter-reason-embed-8b-instructed row does. Leave
    # embed_instruction None and this is the vendor-faithful prompt-free
    # encoder of the plain row - both arms share one rewrite log, because
    # the rewriter input is byte-identical between them.
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

    # --------------------------------------------------------------- scoring

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

        # Query side: encode every rewrite, then mean-pool - fact (2) above.
        # Deliberately uncached: the expensive half (generation) already is,
        # and these vectors are used once per query.
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

        # Document side: straight through the retriever's own per-text vector
        # cache, so documents shared between queries and between tasks are
        # embedded once - exactly as in the standalone row.
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

        # Cosine, not dot product: mean-pooling five unit vectors does not
        # return a unit vector, and EmbeddingProcessor._cosine_similarity
        # normalizes both sides (rank-equivalent to their normalized dot
        # product, and it raises on a zero norm rather than emitting NaNs).
        return self._retriever._cosine_similarity(
            query_embedding, np.asarray(document_embeddings, dtype=float)
        )

    # ------------------------------------------------------------- cache api

    def export_cache(self, path: str) -> None:
        self._init()
        self._retriever.export_cache(path)

    def import_cache(self, path: str) -> None:
        self._init()
        self._retriever.import_cache(path)

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Raw embedding access (NO rewrite) - the retriever backend as-is.

        This is the right call for DOCUMENTS, and for parity checks against
        the standalone row. It is the WRONG call for queries: use
        encode_queries, or the rewrite is silently skipped and you measure
        the bare retriever."""
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
        """One vector per query, WITH the rewrite: each query's n rewrites are
        embedded and mean-pooled, exactly as get_scores does on the query side.

        This exists so a caller that works in vectors rather than scores - a
        FAISS-style sweep over the whole corpus, e.g. scripts/run_dedup.py's
        all-documents regime - can use this model correctly. encode() cannot
        serve that role: it has no way to know a text is a query, so it would
        return bare-retriever vectors under this model's name.

        All rewrites are generated in ONE batched call before any embedding
        (see prefetch_rewrites), so this is not the slow per-query path."""
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
