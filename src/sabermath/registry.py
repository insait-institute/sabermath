"""The SABER-Math model registry: every evaluable model key, its processor
recipe, and its per-model input protocol.

This is the single source of truth for "what models exist and how is each one
built". It used to live inside the reranker CLI script, which meant the table
generators had to import a CLI to learn a model's display category. Nothing
here touches argv, the filesystem, or a results directory - it is a pure
lookup layer over sabermath.processors, so reporting code and run code read
the same definitions.

FIVE REGISTRIES, by how a model is constructed:

    CUSTOM_MODEL_BUILDERS   needs a bespoke processor (rank1, Qwen3-Reranker,
                            ColBERT, ReasonIR, SPLADE, RaDeR, GroupRank,
                            Retro*, the composed rewrite+embed systems)
    GENERIC_MODELS          resolvable from (HF name, use_vllm) alone
    TABLE_MODELS            same, for the models that only appear in the
                            paper's tables
    API_MODELS              closed/hosted embedding APIs
    LEXICAL_MODEL_BUILDERS  BM25 / TF-IDF / Jaccard / Approach Zero

ALL_MODEL_KEYS is the union, and is what `--models` defaults to.

INPUT PROTOCOL. Each model's vendor-documented input envelope (query/document
prefixes and suffixes, per-side API parameters) lives beside its builder and
is applied by EmbeddingProcessor.get_scores per side, before the vector
cache, so an affixed text is simply a new cache key. There is exactly ONE
protocol - the vendor envelopes are always applied. The pre-2026-08-25
envelope-free protocol and its `--legacy` switch were removed on 2026-08-31;
runs made under it are not reproducible from this code, by design.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from .instructions import INSTRUCTIONS
from .processors import (
    Approach0Processor,
    BM25Processor,
    ColBERTProcessor,
    GoogleProcessor,
    GroupRankProcessor,
    INFXRetrieverProcessor,
    JaccardProcessor,
    OpenAIProcessor,
    OpenRouterEmbeddingProcessor,
    Qwen3RerankerVLLMProcessor,
    RaDeRRerankerVLLMProcessor,
    Rank1Processor,
    ReasonIRProcessor,
    ReasonRewriterProcessor,
    RetroStarProcessor,
    RetroStarRewrittenProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
    TfidfProcessor,
    VLLMProcessor,
)
from .schemas import Branch, Task

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

EXPERIMENT_COLBERT_QUERY_LENGTH = 256
EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE = 1280

# Qwen3-Reranker's own model-card default instruction. Used as the p0
# baseline for that family so p0 measures "no task instruction" like every
# other row does; DEFAULT_TASK_INSTRUCTION (the repo's production
# math-specific instruction) is available as prompt key "pm".
VENDOR_QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Models with no vendor-documented free-text instruction mechanism. Their
# p1-p3 rows are CONTROLS (a flat control row is the evidence that the effect
# on genuinely instructable models is real), not instruction-following
# measurements - aggregate_instructions.py reports them in a separate block.
# Deliberately NOT the same thing as INSTRUCTION_EXCLUDED, which hard-errors.
INSTRUCTION_CONTROL_REASONS = {
    "bge-m3": "vendor removed instruction prompting by design",
    "bert-base-uncased": "no instruction mechanism (plain MLM encoder)",
    "roberta-base": "no instruction mechanism (plain MLM encoder)",
    "splade-code-0.6b": "prompts {}; prompt_type picks a top-k budget, not text",
    "splade-code-8b": "prompts {}; prompt_type picks a top-k budget, not text",
    "bm25": "lexical, no instruction mechanism",
    "tf-idf": "lexical, no instruction mechanism",
    "jaccard": "lexical, no instruction mechanism",
    "bm25-no-tok": "lexical, no instruction mechanism",
    "tf-idf-no-tok": "lexical, no instruction mechanism",
    "jaccard-no-tok": "lexical, no instruction mechanism",
    "approach0": "structure search engine, no instruction mechanism",
    "embeddinggemma-300m": "fixed closed-set prompt grammar, free text is not the mechanism",
    "multilingual-e5-large": "fixed query:/passage: prefixes, free text is not the mechanism",
    "jina-embeddings-v5-text-nano": "fixed Query:/Document: prefixes + LoRA task adapter",
    "jina-embeddings-v5-text-small": "fixed Query:/Document: prefixes + LoRA task adapter",
    "gemini-embedding-001": "task_type enum API parameter, no free text",
    "gemini-embedding-2": "task_type enum API parameter, no free text",
    "text-embedding-3-small": "API exposes no instruction parameter at all",
    "text-embedding-3-large": "API exposes no instruction parameter at all",
    "gte-moderncolbert": "[Q]/[D] markers only, no task slot",
    "reason-moderncolbert": "[Q]/[D] markers only, no task slot",
    "rank1-0.5b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-7b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-32b": "no instruction slot; the vendor route is rewriting the query",
    "rank1-32b-bf16": "no instruction slot; the vendor route is rewriting the query",
    "diver-grouprank-32b": "fixed rubric template, no task slot",
    "rader-reranker-7b": "T10 query:/Query:/document: template, no instruction slot",
}
INSTRUCTION_CONTROL_MODELS = frozenset(INSTRUCTION_CONTROL_REASONS)

# ReasonIR. Measured against the official remote-code encoder on 2026-08-25;
# the verdict is archived at results/diagnostics/protocol/reasonir_verdict.json
# (the probe itself is gone - see docs/backend-provenance.md). Its encode() does:
#
#     batch = [instruction + text + embed_eos for text in texts]
#     outputs = self(**inputs)                      # full bidirectional pass
#     if instruction: attention_mask[:, :len(instruction_tokens)] = 0
#     embeddings = self.pooling(last_hidden_state, attention_mask)
#
# so with instruction="" - the document side, and the whole prompt-free
# protocol - it prepends NOTHING. Sending bare text through the production
# vLLM mean pooler therefore reproduces the official vectors to cosine
# 0.9999 (Spearman 1.0 on candidate ranking), while adding an "<|embed|>\n"
# wrapper DROPS that agreement to 0.879: the wrapper tokens end up inside
# vLLM's mean, where the official code has them masked out. Prompt-free is
# the faithful configuration here, not a deviation from one.
#
# For the instructed arms the official mechanism is "prepend
# <|user|>\n{task}\n<|embed|>\n, then exclude exactly those tokens from the
# pool". vLLM cannot exclude them, and both vLLM approximations land at
# cosine ~0.80 / Spearman 0.93 against the official encoder. The exact route
# is ReasonIRProcessor with per-side encode kwargs, which
# EmbeddingProcessor.get_scores now supports directly:
REASONIR_QUERY_INSTRUCTION_SLOT = "<|user|>\n{instruction}\n<|embed|>\n"

# Tokenizer EOS expected by the RaDeR bi-encoder envelope below; asserted
# against the checkpoint before the envelope shipped (docs/protocol.md).
RADER_EXPECTED_EOS = "<|im_end|>"

# Models needing a custom ModelProcessor (not resolvable from a bare HF name).
# rank1-32b's Processor CAN take tensor_parallel_size > 1 (vLLM generation,
# actually GPU-hungry at 32B), but a direct diagnostic re-run confirmed that
# doing so (tensor_parallel_size=2, matched to a 2-GPU allocation) silently
# corrupts its relevance scores - see the note in Rank1Processor._init().
# Hardcoded to 1 here regardless of what --tensor-parallel-size/-gpus is
# passed, until/unless that's re-investigated (e.g. via data parallelism -
# two independent single-GPU engines splitting the query set - instead of
# tensor parallelism). The other models below are single-process HF/pylate
# calls that don't shard across GPUs at all, so --tensor-parallel-size is
# simply ignored for them too (see _run_one).
# RaDeR bi-encoder retrievers (the size-ablation family from the same
# collection - 3B/7B/14B all share the recipe below; each card's own vLLM
# serving snippet confirms `pooling_type=LAST, normalize=true` for every
# size).
RADER_BIENCODER_MODELS = {
    "rader-3b": "Raderspace/RaDeR_Qwen25_3B_NuminaMath_MATH_allquerytypes",
    "rader-7b": "Raderspace/RaDeR_Qwen25-7B_NuminaMath_MATH_allquerytypes",
    "rader-14b": "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes",
}
# NOTE the repo-name inconsistency above is real and verified against the HF
# API: the 3B uses an underscore ("Qwen25_3B"), the 7B/14B a hyphen
# ("Qwen25-7B"/"Qwen25-14B").


def _build_rader_biencoder_vllm(model_name: str):
    """PRODUCTION path for the RaDeR bi-encoders since 2026-08-20: vLLM with
    the exact recipe validated against raw-HF last-token embeddings by
    the 2026-08-20 vLLM feasibility sweep (Spearman 0.9999-1.0 vs a fixed
    raw-HF last-token reference; verdicts archived in
    results/diagnostics/vllm_feasibility/summary.json). Every knob below is
    load-bearing:
    - pooling_type=LAST per each card's own vLLM serving snippet;
      normalize=False (not the card's normalize=true) because the production
      protocol mean-averages UN-normalized chunk vectors before the final
      cosine - see docs/backend-provenance.md;
    - dtype=bfloat16 (since 2026-08-21): bf16 is RaDeR's TRAINING precision
      (their repo trains with --bf16; the checkpoint's torch_dtype float32
      is just storage format) and was validated FEASIBLE against the fixed
      raw-HF references (Spearman 0.9978-0.9994, |dNDCG@10| <= 0.019) while
      running up to ~16x faster end-to-end than fp32 on the 14B. The
      earlier "reduced precision wrecks rankings (Spearman 0.19-0.39)"
      finding that had forced fp32 was CONFOUNDED - it was measured against
      the chat-template-corrupted reference, and that mismatch was entirely
      the reference bug, not dtype. NOTE on provenance: the published
      2026-08-20 full nDCG runs were produced in fp32; the bf16 delta is
      below the bootstrap CI half-widths;
    - max_model_len=2080 (not 2048): chunk_to_context produces chunks of UP
      TO exactly 2048 tokens, and prompt_len == max_model_len sat right on
      the boundary where the engine wedge below was observed - the margin
      removes the boundary case outright;
    - enable_prefix_caching/enable_chunked_prefill=False: with vLLM's
      defaults (both True) the first full-benchmark rader runs WEDGED
      mid-statement-full (2026-08-20, jobs 727753-5 + the 730036 retry):
      client blocked forever in core_client.get_output, EngineCore alive but
      busy-looping with nothing scheduled, GPU at 0% until the idle reaper
      killed the job. Non-deterministic (retry passed the previously-hung
      query, then wedged two queries later), first appearing exactly where
      multi-chunk (2048-token) prompts start. The bert/e5 engines - which
      auto-run with both features off - completed every run cleanly, and
      neither feature helps embedding workloads (our EmbeddingProcessor
      caches per-text results anyway), so both are disabled rather than
      diagnosed further upstream."""
    return VLLMProcessor.from_huggingface(
        model_name,
        pooler_config={"pooling_type": "LAST", "normalize": False},
        dtype="bfloat16",
        max_model_len=2080,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )


def _build_inf_retriever_processor(model_name: str = "infly/inf-retriever-v1-pro"):
    """INF-Retriever-v1-Pro (7B, gte-Qwen2 lineage): auto-config
    SentenceTransformers load. The repo ships the full ST stack
    (modules.json: Transformer + lasttoken Pooling + Normalize) so unlike
    the RaDeR bi-encoders no hand-built module stack is needed - this IS
    the author-blessed path.

    Deliberate exception to the 2026-08-20 vLLM-default policy: the repo's
    bundled modeling_qwen.py runs BIDIRECTIONAL attention when used as an
    embedder (Qwen2Model.forward's is_causal defaults to False - checked in
    the repo source; same design as Alibaba's gte-Qwen2-7B-instruct this
    lineage descends from), while vLLM's stock Qwen2Model is causal. A
    candidate vLLM recipe (hf_overrides is_causal=False + LAST pooling) is
    recorded in docs/backend-provenance.md - switch this builder
    only after a FEASIBLE verdict is measured; running it causal-by-default would
    be exactly the silent wrong-attention/wrong-pooling class of bug that
    burned the RaDeR family.

    Load-bearing details:
    - trust_remote_code=True matters twice over: the bidirectional modeling
      AND the repo's tokenization_qwen.py, whose add_eos_token=true appends
      <|endoftext|> so "lasttoken" pooling lands on a constant summary
      position (a vanilla Qwen2Tokenizer silently ignores add_eos_token).
    - model_kwargs torch_dtype="auto" -> fp16, the checkpoint's declared
      dtype; ST's default load would silently compute in fp32, against the
      2026-08-21 precision-fairness policy (no model computes in fp32).
    - the tokenizer ships a chat template, so the sentence-transformers
      >=5.7 "message"-modality latch that chat-template-wrapped every RaDeR
      input (see docs/backend-provenance.md) applies here too; strip it post-init - a no-op if ST didn't infer it.
    - prompt-less on purpose: the repo defines a "query" instruction prompt
      (config_sentence_transformers.json) but default_prompt_name is null,
      and the benchmark protocol runs every model without instruction
      prefixes (instruction prompting is the team's separate experiment).
    - no chunking: max_seq_length is 32768 (sentence_bert_config.json) and
      essentially no benchmark document comes near it (only 0.32% exceed
      even 2048 tokens), so ST's silent truncation at 32768 is a no-op in
      practice.

    The actual ST construction lives in
    sabermath.processors.infx_retriever_processor.build_inf_retriever_st -
    promoted there (2026-08-21) so the INF-X-Retriever composition reuses
    the byte-identical retriever backend and the standalone-vs-INF-X
    comparison isolates the query rewrite alone.
    """
    from sabermath.processors import build_inf_retriever_st

    return STProcessor(build_inf_retriever_st(model_name), model_name)


CUSTOM_MODEL_BUILDERS = {
    "rank1-32b": lambda tp: Rank1Processor(tensor_parallel_size=1),
    # rank1 size ablations - same processor/prompt/scoring as rank1-32b,
    # just a different HF repo. tensor_parallel_size stays pinned to 1 (see
    # the rank1-32b note above; >1 would be pointless at these sizes anyway).
    "rank1-7b": lambda tp: Rank1Processor("jhu-clsp/rank1-7b"),
    "rank1-0.5b": lambda tp: Rank1Processor("jhu-clsp/rank1-0.5b"),
    # dtype ablation, NOT a separate model. rank1-32b's checkpoint is
    # bfloat16 and the production spec above pins vLLM to float16 (inherited
    # from the model card), so every production run logs "Casting
    # torch.bfloat16 to torch.float16". fp16's exponent range is far narrower
    # than bf16's, so this key runs the identical model at the checkpoint's
    # native precision to measure what that downcast costs. Same prompt, same
    # scoring, same tensor_parallel_size=1 - dtype is the ONLY difference.
    "rank1-32b-bf16": lambda tp: Rank1Processor(
        tensor_parallel_size=1, dtype="bfloat16"
    ),
    # Qwen3-Reranker family: vLLM-backed since 2026-08-20 (official model-card
    # recipe; verified FEASIBLE vs the HF path, Spearman >= 0.999 - see
    # results/diagnostics/vllm_feasibility/summary.json).
    "qwen3-reranker-8b": lambda tp: Qwen3RerankerVLLMProcessor(),
    "qwen3-reranker-4b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-4B"
    ),
    "qwen3-reranker-0.6b": lambda tp: Qwen3RerankerVLLMProcessor(
        "Qwen/Qwen3-Reranker-0.6B"
    ),
    "gte-moderncolbert": lambda tp: ColBERTProcessor("lightonai/GTE-ModernColBERT-v1"),
    "reason-moderncolbert": lambda tp: ColBERTProcessor("lightonai/Reason-ModernColBERT"),
    # reasonir-8b: the official remote-code path (reverted from vLLM on
    # 2026-08-25). vLLM cannot reproduce the one thing that makes ReasonIR's
    # instruction mechanism work - its encode() runs a full bidirectional
    # pass over "instruction + text + embed_eos" and then zeroes the
    # instruction positions in the POOLING mask, which a stock MEAN pooler
    # has no way to express. Prompt-free the two agree to cosine 0.9999, so
    # this also restores the provenance of the published number, which
    # predates the vLLM switch. See REASONIR_INSTRUCTION_NOTE.
    "reasonir-8b": lambda tp: ReasonIRProcessor(),
    # p0-ONLY vLLM variant of reasonir-8b, added 2026-08-26. A SEPARATE key
    # rather than a change to the entry above, because the ST path must stay
    # for the instructed arms and for the published number's provenance.
    #
    # ReasonIR declares architectures: ["ReasonIRModel"], which vLLM does not
    # register - hence the hf_overrides redirect onto vLLM's own bidirectional
    # Llama, which is now known correct (it reproduced a genuinely
    # bidirectional reference for llama-embed-nemotron-8b to median |delta|
    # 6.9e-05 with zero verdict flips).
    #
    # MEASURED, not assumed (math-vs-word, 969 targets; see
    # docs/backend-provenance.md): against the ST reference built under
    # this model's own transformers==4.47.1 pin, this recipe agrees to
    # median |delta| 0.0012 with 2/969 verdict flips and an identical
    # math-vs-word statistic (66.46% both). That independently confirms the
    # "agree to cosine 0.9999 prompt-free" claim in the reasonir-8b comment
    # above - worth stating because the feasibility sweep that claim cites
    # is archived as data (results/diagnostics/vllm_feasibility/), not code.
    #
    # STRICTLY p0. See INSTRUCTION_EXCLUDED below: vLLM cannot express this
    # model's instruction mechanism (its encode() runs a bidirectional pass
    # over "instruction + text + embed_eos" and then zeroes the instruction
    # positions in the POOLING mask, which a stock MEAN pooler has no way to
    # represent), which is exactly why reasonir-8b was reverted off vLLM on
    # 2026-08-25. Instructed arms hard-error rather than silently returning
    # numbers from a mechanism that was not applied.
    "reasonir-8b-vllm": lambda tp: VLLMProcessor.from_huggingface(
        "reasonir/ReasonIR-8B",
        hf_overrides={
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        pooler_config={"pooling_type": "MEAN", "normalize": True},
    ),
    "splade-code-8b": lambda tp: SpladeProcessor(),
    # NOTE the 0.6B repo really is named "splade-code-06B" (no dot) -
    # verified against the HF API; "naver/splade-code-0.6B" does not exist.
    "splade-code-0.6b": lambda tp: SpladeProcessor("naver/splade-code-06B"),
    "rader-3b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-3b"]
    ),
    "rader-7b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-7b"]
    ),
    "rader-14b": lambda tp: _build_rader_biencoder_vllm(
        RADER_BIENCODER_MODELS["rader-14b"]
    ),
    # RaDeR's pointwise cross-encoder reranker: vLLM-backed since 2026-08-20
    # (one-off LoRA merge; verified FEASIBLE, Spearman 0.9968 - see
    # RaDeRRerankerVLLMProcessor).
    "rader-reranker-7b": lambda tp: RaDeRRerankerVLLMProcessor(),
    # Diver's groupwise generative reranker (vLLM) - see GroupRankProcessor.
    # Unlike rank1-32b, tensor parallelism has no known score-corruption
    # history for this model, so --tensor-parallel-size is honored here; the
    # default single-GPU allocation still fits the 32B in bf16 on one H200.
    "diver-grouprank-32b": lambda tp: GroupRankProcessor(
        tensor_parallel_size=max(1, tp)
    ),
    # INF-Retriever-v1-Pro: SentenceTransformers-backed ON PURPOSE (the one
    # bi-encoder exception to the 2026-08-20 vLLM-default rollout) -
    # bidirectional remote-code attention with no VALIDATED vLLM path yet.
    # See _build_inf_retriever_processor.
    "inf-retriever-v1-pro": lambda tp: _build_inf_retriever_processor(),
    # INF-X-Retriever: the full infly SYSTEM the standalone model above was
    # built for - inf-query-aligner rewrites each query (official prompt +
    # the checkpoint's own sampling params, made reproducible via
    # per-query-content seeding), then the REWRITE is embedded with the
    # official instruct prefix against prefix-less documents, through the
    # byte-identical retriever backend as the standalone entry (so the two
    # rows isolate the rewrite+prefix effect). Recipe verified against the
    # team's own code (github.com/yaoyichen/INF-X-Retriever) - see
    # INFXRetrieverProcessor's module docstring for the full audit.
    # Deliberately NOT in the feasibility sweep: it compares deterministic
    # scoring paths of single models; this entry's
    # retriever half is already covered by inf-retriever-v1-pro's candidate
    # there, and its generative half has no reference to regress against.
    # Two 7B models co-resident on one GPU (~30GB in bf16+fp16) - fine on
    # an H200, no tensor parallelism. The rewrite log makes every generated
    # rewrite inspectable after the run (and warm-starts resumes; the
    # per-query seeding makes regeneration byte-identical either way) -
    # kept under the default save dir so the region sync ships it back to
    # the canonical host with the results.
    "inf-x-retriever": lambda tp: INFXRetrieverProcessor(
        rewrite_log_path="results/evaluation/.rewrites/inf-x-retriever.json"
    ),
    # Retro* (VectorSpaceLab): a generative POINTWISE reranker whose score is
    # an integer the model writes into a <score> tag, not a logprob - see
    # RetroStarProcessor for the transcription of their released evaluation
    # code and for the vendor relevance definition used here (their `aops`
    # triple, whose query/document shapes are exactly statement-full's).
    #
    # The 8B is the size the team's own released BRIGHT scripts run; the 32B
    # is the largest checkpoint of the family. Tensor parallelism is honored
    # for both (they are plain vLLM generation, with none of rank1-32b's
    # score-corruption history), and the 32B WANTS it: this is by far the
    # most expensive entry in the registry - ~150 candidates x ~1000 queries
    # x up to 4096 generated tokens, i.e. a full generative pass per PAIR.
    # Budget it like rank1-32b (--time=3-0, resume-on-timeout) and give it
    # every GPU on the node.
    "retro-star-32b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-32b-0928", tensor_parallel_size=max(1, tp)
    ),
    "retro-star-8b": lambda tp: RetroStarProcessor(
        "ljw13/retro-star-qwen3-8b-0928", tensor_parallel_size=max(1, tp)
    ),
    # Reason-Rewriter + Reason-Embed-Qwen3-8B: the BGE-Reasoner "rewrite,
    # then embed" SYSTEM, standing to reason-embed-qwen3-8b exactly as
    # inf-x-retriever stands to inf-retriever-v1-pro - same embedding
    # backend, same prompt-free protocol, so the pair of rows isolates the
    # rewrite. Five sampled rewrites per query with their embeddings
    # mean-pooled (the shape of the team's own released rewrite artifact),
    # made reproducible by per-query-content seeding. Two vLLM engines share
    # the GPU, so each gets an explicit memory fraction; no tensor
    # parallelism. See ReasonRewriterProcessor for the full protocol audit.
    "reason-rewriter-reason-embed-8b": lambda tp: ReasonRewriterProcessor(
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        )
    ),
    # Same composed system with the LLaMA-3.1 Reason-Embed as the retriever
    # half, so "does the rewrite help this embedder?" can be asked of both
    # Reason-Embed checkpoints rather than only the Qwen3 one. Added
    # 2026-08-28 for the math-vs-word rewritten arm.
    #
    # It SHARES the Qwen3 row's rewrite log, which is correct rather than
    # convenient: the rewriter half, its task string and the recipe
    # fingerprint are identical, and _load_rewrite_log validates exactly
    # those three before trusting an entry. Only the embedder differs, and
    # the log holds no embeddings. So every query the Qwen3 row has already
    # rewritten is a cache hit here, and the 7B generator never reloads for it.
    #
    # max_model_len=40960 mirrors the standalone reason-embed-llama-3.1-8b
    # spec above; without it vLLM would take llama-3.1's 131072 default and
    # the two rows would no longer be preprocessing-identical.
    "reason-rewriter-reason-embed-llama-3.1-8b": (
        lambda tp: ReasonRewriterProcessor(
            retriever_name="hanhainebula/reason-embed-llama-3.1-8b-0928",
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            ),
            retriever_init_kwargs={"max_model_len": 40960},
        )
    ),
    # The SAME system with the instruction reaching BOTH halves. On the row
    # above, --instructions pN only ever reaches the REWRITER: the harness
    # wraps the instruction into the query text, _rewrite() consumes that
    # wrapped text, and the query side then embeds the REWRITES - so the
    # encoder runs prompt-free under every prompt key. This key reapplies the
    # same wrap to each rewrite just before embedding (via the
    # embed_instruction kwarg below and
    # ReasonRewriterProcessor._texts_to_embed), so pN steers the generator AND
    # the encoder.
    #
    # That makes it the missing third arm: standalone reason-embed-qwen3-8b
    # instructs the ENCODER alone, the plain composed row instructs the
    # REWRITER alone, and this one instructs both. It shares the plain row's
    # rewrite log on purpose - the rewriter input is byte-identical between
    # the two composed arms, so every cached p1/p2 query stays a cache hit and
    # the delta between the rows isolates the encoder-side instruction alone.
    # At p0 the two composed rows are the same measurement by construction:
    # INSTRUCTIONS["p0"] is None, so there is no wrap to reapply.
    "reason-rewriter-reason-embed-8b-instructed": (
        lambda tp: ReasonRewriterProcessor(
            rewrite_log_path=(
                "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
            )
        )
    ),
    # Reason-Rewriter feeding the RERANKER instead of the retriever: Retro*
    # scores each pair against the rewritten query rather than the problem
    # statement. NOT the vendor's cascade - theirs reranks the ORIGINAL query
    # and uses the rewrite only for retrieval, which on this benchmark's fixed
    # 150-candidate sets would reproduce retro-star-8b exactly. See
    # RetroStarRewrittenProcessor before reporting this number.
    #
    # It reads the SAME rewrite log the reason-rewriter-reason-embed-8b row
    # wrote, so every query is a cache hit and the 7B generator never loads.
    "retro-star-8b-rewritten": lambda tp: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-8b-0928",
        tensor_parallel_size=max(1, tp),
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
    ),
    # Same rewrite-as-the-reranker's-query construction at 32B. Read the
    # warning on retro-star-8b-rewritten first: this is NOT the vendor's
    # cascade, and its number must not be reported as one.
    #
    # Two differences from the 8B entry, both because of the size. The
    # reranker gets 0.85 of the GPU instead of 0.55 - a 32B in bf16 is ~64GB
    # of weights, and at 0.55 there is almost no KV cache left, which with
    # ~11k-token prompts (a ~4k-token rewrite plus the document) would leave
    # one or two candidates in flight. That budget only works because every
    # rewrite is already cached and the 7B generator never loads, so
    # require_cached_rewrites makes the alternative a clear error rather than
    # an OOM. Run this SHARDED like plain retro-star-32b (--query-shards);
    # longer prompts make it slower per query than that row.
    "retro-star-32b-rewritten": lambda tp: RetroStarRewrittenProcessor(
        "ljw13/retro-star-qwen3-32b-0928",
        tensor_parallel_size=max(1, tp),
        rewrite_log_path=(
            "results/evaluation/.rewrites/reason-rewriter-reason-embed-8b.json"
        ),
        gpu_memory_utilization=0.85,
        require_cached_rewrites=True,
    ),
}

# Custom models where --tensor-parallel-size is actually honored (everything
# else is either pinned to 1 for correctness - rank1-32b - or a single-GPU
# HF/PyLate call that can't shard at all).
CUSTOM_MODELS_USE_TP = {
    "diver-grouprank-32b",
    "retro-star-32b",
    "retro-star-8b",
    "retro-star-8b-rewritten",
    "retro-star-32b-rewritten",
}

# Per-model scores_kwargs for CUSTOM_MODEL_BUILDERS entries (forwarded to
# processor.get_scores(**scores_kwargs) - see _run_one). The rader family
# keeps chunk_to_context/context_length=2048 (the paper's preprocessing
# protocol - only 0.32% of documents exceed 2048 tokens, but the numbers must
# stay preprocessing-identical across backends and sizes). batch_size was
# dropped when the family moved to vLLM (2026-08-20): it was an ST-side OOM
# guard; vLLM schedules/batches internally.
#
# The query_prompt/document_prompt/query_suffix/document_suffix entries are
# the vendor-documented input envelope for each model (see
# docs/protocol.md): EmbeddingProcessor.get_scores applies
# them per side, before the vector cache, so an affixed text is simply a new
# cache key.
#
# RaDeR: their retrievers.py builds "query: {instruction}{query}<|im_end|>"
# and "document: {doc}<|im_end|>" with last-token pooling; we sent raw text
# with neither marker nor EOS.
_RADER_BIENCODER_SCORES_KWARGS = {
    "chunk_to_context": True,
    "context_length": 2048,
    "query_prompt": "query: ",
    "document_prompt": "document: ",
    "query_suffix": RADER_EXPECTED_EOS,
    "document_suffix": RADER_EXPECTED_EOS,
}
# ReasonIR gets no text envelope: its own encode() adds nothing when the
# instruction is empty. The instructed arms go through per-side encode
# kwargs instead - see _experiment_scores_kwargs.
CUSTOM_MODEL_SCORES_KWARGS = {
    "rader-3b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-7b": dict(_RADER_BIENCODER_SCORES_KWARGS),
    "rader-14b": dict(_RADER_BIENCODER_SCORES_KWARGS),
}

# Already supported through evaluate()'s generic HuggingFace path - no custom
# processor needed, just the right (model name, use_vllm) pair.
# tensor_parallel_size is forwarded via init_kwargs since
# VLLMProcessor.from_huggingface passes **kwargs straight through to
# vllm.LLM(...).
#
# Every generic here is vLLM-backed since 2026-08-20 (previously only
# qwen3-embedding-8b was; the others ran through the auto-detected
# SentenceTransformers path). The Qwen3-Embedding derivatives get an
# explicit LAST+normalize pooler override as a plain dict (VLLMProcessor
# converts it version-tolerantly) - deterministic rather than trusting arch
# defaults, and verified FEASIBLE against the old ST path (Spearman 0.999-1.0,
# results/diagnostics/vllm_feasibility/summary.json). trust_remote_code is
# VLLMProcessor.from_huggingface's default, so no init kwarg needed for it;
# the old ST-only model_kwargs/torch_dtype init kwargs would crash vllm.LLM
# and are gone with the backend.
GENERIC_MODELS = {
    "qwen3-embedding-8b": {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True},
    "reason-embed-qwen3-8b": {
        "model": "hanhainebula/reason-embed-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    # Three more checkpoints from the same Reason-Embed release (0928), added
    # 2026-08-28 as a backbone/size ablation around reason-embed-qwen3-8b:
    # "basic" is the same Qwen3-8B backbone trained without the release's
    # reasoning-augmented data, -qwen3-4b is the size sibling, and
    # -llama-3.1-8b swaps the backbone at matched size.
    #
    # All three are configured IDENTICALLY to reason-embed-qwen3-8b above, on
    # purpose - an ablation is only readable if preprocessing is identical
    # across its arms - and the configs justify it: every repo in the family
    # ships modules.json = [Transformer, Pooling], 1_Pooling with
    # pooling_mode_lasttoken=true and nothing else, and
    # similarity_fn_name=cosine, so the explicit LAST+normalize pooler
    # override is the vendor stack in all four cases.
    #
    # No dtype pin, matching the 8B sibling: every one of these repos declares
    # torch_dtype float32, and vLLM's "auto" downcasts a float32 pooling model
    # to float16 (_resolve_auto_dtype prefers fp16 for pooling models, then
    # "Downcasting float32 to float16"), so all four arms land on fp16 without
    # being told to. This is the SAME situation diver-retriever-0.6b needed a
    # pin for and the opposite conclusion: there the sibling configs disagreed
    # (one declared bf16, one declared nothing), here they all agree, so a pin
    # would only create a split.
    #
    # No vendor query envelope either, again matching the 8B sibling.
    # config_sentence_transformers.json in all four repos declares
    # prompts.query = "Instruct: Given a query, retrieve documents that can
    # help answer the query.\nQuery: " with an empty document prompt - that is
    # exactly the shape the instruction experiment supplies through
    # INSTRUCTION_TEMPLATES["canonical"], so the production (p0) arm stays
    # prompt-free here and the prompt is studied there instead.
    "reason-embed-basic-qwen3-8b": {
        "model": "hanhainebula/reason-embed-basic-qwen3-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "reason-embed-qwen3-4b": {
        "model": "hanhainebula/reason-embed-qwen3-4b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    # max_model_len is the ONE deliberate difference from its siblings, and it
    # is result-neutral: this repo is a Llama-3.1 backbone, so its config
    # advertises 131072 positions where the Qwen3 arms advertise 40960. vLLM
    # disables chunked prefill for pooling models, so an unpinned 131072 makes
    # it size max_num_batched_tokens (and the profiling run) for a 131k-token
    # batch this benchmark can never produce - the longest statement-full
    # document is 15518 characters (~6k tokens) and p99 is 3896, i.e. nothing
    # here is truncated at 40960 by either arm. Pinning it to the Qwen3 arms'
    # window keeps the effective context of the ablation identical too. The
    # vendor's own BRIGHT script (evaluation_scripts/eval_bright_short.sh in
    # every repo of the family) truncates at 8192 on both sides, well below
    # this, so 40960 is not a length the vendor relies on either.
    "reason-embed-llama-3.1-8b": {
        "model": "hanhainebula/reason-embed-llama-3.1-8b-0928",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "max_model_len": 40960,
        },
    },
    # yale-nlp/RTriever-4B: a Qwen3-Embedding-4B finetune for
    # reasoning-intensive retrieval. Same architecture family (Qwen3, LAST
    # pooling, cosine) as the entries above, so it gets the same explicit
    # pooler override; its repo also ships a 2_Normalize module, which that
    # override reproduces on the vLLM side. dtype is left to "auto" (the
    # config declares bfloat16, so no downcast question arises here at all)
    # and no query prompt is applied: the repo declares the stock
    # Qwen3-Embedding "Instruct: ...\nQuery:" prompt, and this registry runs
    # that whole family prompt-free in the production arm (see
    # qwen3-embedding-8b and the reason-embed block above), with the
    # instruction dimension studied through --instructions instead.
    "rtriever-4b": {
        "model": "yale-nlp/RTriever-4B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    "diver-retriever-4b": {
        "model": "AQ-MedAI/Diver-Retriever-4B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
        },
    },
    # Size-ablation sibling of diver-retriever-4b - identical init_kwargs so
    # the size comparison stays preprocessing-clean. dtype is pinned to
    # bfloat16 (2026-08-21): this repo's config declares no torch_dtype, so
    # vLLM's "auto" silently ran it in fp16 while the 4B sibling (bf16
    # config) ran bf16 - an inconsistent precision split within one
    # size-ablation family. Explicit bf16 validated against the frozen
    # raw-HF reference at Spearman 0.9989 (identical to the fp16 verdict).
    "diver-retriever-0.6b": {
        "model": "AQ-MedAI/Diver-Retriever-0.6B",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "LAST", "normalize": True},
            "dtype": "bfloat16",
        },
    },
}
# rader-14b moved out of GENERIC_MODELS and into CUSTOM_MODEL_BUILDERS below -
# see _build_rader_14b_processor for why.
GENERIC_MODELS_USE_TP = {"qwen3-embedding-8b"}  # vLLM-backed -> tp actually used

# Models that need a GPU-side processor from this repo's own pipeline. Kept
# as its own name because the dependency-isolation rules (one conda env per
# model family) apply to exactly these keys, not to the API/lexical ones.
PIPELINE_MODEL_KEYS = list(CUSTOM_MODEL_BUILDERS) + list(GENERIC_MODELS)

_CHUNK512_SCORES_KWARGS = {"chunk_to_context": True, "context_length": 512}

TABLE_MODELS = {
    "qwen3-embedding-4b": {"model": "Qwen/Qwen3-Embedding-4B", "use_vllm": True},
    "qwen3-embedding-0.6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "use_vllm": True,
    },
    "harrier-oss-v1-270m": {
        "model": "microsoft/harrier-oss-v1-270m",
        "use_vllm": True,
    },
    "harrier-oss-v1-0.6b": {
        "model": "microsoft/harrier-oss-v1-0.6b",
        "use_vllm": True,
    },
    "harrier-oss-v1-27b": {
        "model": "microsoft/harrier-oss-v1-27b",
        "use_vllm": True,
    },
    "bge-m3": {"model": "BAAI/bge-m3", "use_vllm": True},
    # NOT given init_kwargs, and deliberately NOT moved off vLLM - both were
    # tried on 2026-08-26 and both were wrong. Full trail, because the
    # conclusion is counter-intuitive:
    #
    # Switching this model to vLLM changed its math-vs-word statistic by
    # -8.98pp (70.79% -> 61.82%, 107/969 verdict flips), which looked like a
    # vLLM defect. It is the opposite: vLLM is CORRECT and the previous
    # SentenceTransformers numbers were silently wrong.
    #
    # This repo declares architectures: ["LlamaBidirectionalModel"] with an
    # auto_map onto its own llama_bidirectional_model.py, which makes the
    # encoder bidirectional through ONE effective hook: an override of
    # LlamaModel._update_causal_mask() that returns no mask. (Its other hook,
    # layer.self_attn.is_causal = False, is inert - transformers never
    # forwards is_causal to the attention interface.) That override also
    # ASSERTS the attention impl is flash_attention_2 or eager.
    #
    # transformers 5.x DELETED _update_causal_mask: LlamaModel.forward calls
    # masking_utils.create_causal_mask directly. So under transformers 5.16.1
    # - which an unpinned install now resolves to - the override is never
    # called, the assert never fires, and the model runs CAUSAL while
    # reporting nothing. Every ST number for this model produced that way is
    # a causal run of a bidirectional model.
    #
    # Measured (experiments/math-vs-word, 969 targets, same subset):
    #   ST tf5.16.1 sdpa   (causal, as published)   78.95%
    #   vLLM fp32                                   68.02%
    #   ST tf4.51.3 eager  (genuinely bidirectional) 68.02%
    #   bidirectional ST vs vLLM : median |d| 8.2e-05, max 4.4e-04, 0 flips
    #   bidirectional ST vs published ST: median |d| 8.2e-02, 29 flips
    # i.e. vLLM reproduces the bidirectional reference to kernel precision.
    #
    # dtype was also ruled out: pinning float32 left the gap to the causal ST
    # run at median 0.0884 (bf16 gave 0.0889), so precision explained none of
    # it. No dtype pin is kept - vLLM's "auto" reads bfloat16 from the config,
    # and the bf16 and fp32 runs give the same 61.82%.
    #
    # DO NOT "fix" this by setting use_vllm: False without also pinning
    # transformers <5 AND forcing attn_implementation - ST alone silently
    # regresses to causal. Any change here has to be re-verified against a
    # raw-HF bidirectional reference - see docs/backend-provenance.md.
    "llama-embed-nemotron-8b": {
        "model": "nvidia/llama-embed-nemotron-8b",
        "use_vllm": True,
    },
    "kalm-embedding-gemma3-12b-2511": {
        "model": "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "use_vllm": True,
    },
    # EmbeddingGemma is a prompt-template-trained model: its documented
    # grammar is a fixed closed set, not free text.
    "embeddinggemma-300m": {
        "model": "google/embeddinggemma-300m",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "task: search result | query: ",
            "document_prompt": "title: none | text: ",
        },
    },
    # jina-v5 ships prompts {"query": "Query: ", "document": "Document: "}
    # with default_prompt_name="document" - so the prompt-less ST path used
    # to prepend "Document: " to QUERIES as well as documents.
    "jina-embeddings-v5-text-small": {
        "model": "jinaai/jina-embeddings-v5-text-small",
        "use_vllm": True,
        "scores_kwargs": {
            "query_prompt": "Query: ",
            "document_prompt": "Document: ",
        },
    },
    "jina-embeddings-v5-text-nano": {
        "model": "jinaai/jina-embeddings-v5-text-nano",
        "use_vllm": False,
        "init_kwargs": {
            "trust_remote_code": True,
            "device": "cuda",
            "model_kwargs": {"dtype": "bfloat16", "default_task": "retrieval"},
        },
        "scores_kwargs": {
            "query_prompt": "Query: ",
            "document_prompt": "Document: ",
        },
        "disable_st_default_prompt": True,
    },
    # vLLM CONFIRMED CORRECT for this family (2026-08-26), and the pre-2026-08-26
    # SentenceTransformers numbers were the broken ones. Moving Octen-8B to vLLM
    # shifted its math-vs-word statistic +6.60pp (48.30% -> 54.90%, 136/969
    # verdict flips), which looked like a vLLM regression. It is not:
    #
    #   * dtype ruled out - fp32 vLLM matches bf16 vLLM to median 6.5e-04
    #     (2/969 flips), while the gap to ST stayed at 0.069;
    #   * pooling mode ruled out - vLLM matches ST[lasttoken] far better than
    #     ST[mean] (0.56) or ST[cls] (0.10), i.e. LAST is resolved correctly
    #     from the sentence_transformers config;
    #   * tokenization ruled out CONCLUSIVELY - the two backends emit
    #     byte-identical input ids, EOS 151643 included, and re-embedding
    #     through vLLM with explicit prompt_token_ids changes nothing
    #     (0.70380 via text vs 0.70351 via ids);
    #   * padding/batching ruled out - encoding one text at a time instead of
    #     in a padded batch changes nothing (0.99984 either way).
    #
    # The cause was the OLD loader. math-vs-word's _build_octen_processor
    # HAND-BUILT a [Transformer, Pooling(lasttoken), Normalize] stack, on the
    # premise that the generic SentenceTransformer load crashes on this repo's
    # 2_Normalize config. Measured on the same 14 probes:
    #
    #     hand-built ST stack vs vLLM : cosine 0.704
    #     GENERIC ST load     vs vLLM : cosine 0.99978
    #
    # The generic load works ON sentence-transformers 5.7.0 and agrees with
    # vLLM to 2e-04. It is version-dependent: on 6.0.0 it raises
    # "Normalize.__init__() got an unexpected keyword argument
    # 'normalize_embeddings'" and there is then NO correct ST option at all,
    # since the hand-built fallback is the broken one (0.68 there). So the hand-built imitation of the vendor stack was the
    # outlier, not vLLM - the same shape of finding as llama-embed-nemotron-8b
    # above, reached by a different route.
    #
    # Cross-check: Qwen/Qwen3-Embedding-4B, same architecture family, loaded
    # GENERICALLY, agrees with vLLM at 0.9998 - so vLLM's Qwen3 embedding path
    # is sound in general and this was never an Octen-specific vLLM problem.
    #
    # The finding, and how it was established, are in
    # docs/backend-provenance.md.
    "octen-embedding-4b": {"model": "Octen/Octen-Embedding-4B", "use_vllm": True},
    "octen-embedding-8b": {"model": "Octen/Octen-Embedding-8B", "use_vllm": True},
    "roberta-base": {
        "model": "FacebookAI/roberta-base",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": False},
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
    "bert-base-uncased": {
        "model": "google-bert/bert-base-uncased",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": False},
            "export_clean": True,
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
    # DELIBERATELY prompt-free, against the model card. E5 documents
    # "query: " / "passage: " as required ("otherwise you will see a
    # performance degradation"), but measured on this benchmark those
    # prefixes COST 0.028-0.080 nDCG, consistently, at p0, p1 and p2, on both
    # tasks that use problem+solution documents - and the effect is not a
    # chunking artifact (see docs/protocol.md). We report our own
    # prompt-free configuration and disclose the deviation.
    "multilingual-e5-large": {
        "model": "intfloat/multilingual-e5-large",
        "use_vllm": True,
        "init_kwargs": {
            "pooler_config": {"pooling_type": "MEAN", "normalize": True},
            "max_model_len": 512,
        },
        "scores_kwargs": dict(_CHUNK512_SCORES_KWARGS),
    },
}

API_MODELS = {
    "gemini-embedding-001": ("google", "gemini-embedding-001"),
    "gemini-embedding-2": ("google", "gemini-embedding-2"),
    "text-embedding-3-small": ("openrouter", "text-embedding-3-small"),
    "text-embedding-3-large": ("openrouter", "text-embedding-3-large"),
}

# Gemini's retrieval mechanism is the task_type enum, not a text prefix; it
# reaches embed_content(config=EmbedContentConfig(task_type=...)) through
# EmbeddingProcessor's per-side encode kwargs.
#
# gemini-embedding-001 ONLY. Live-probed 2026-08-25 against the real API:
# on -001, RETRIEVAL_QUERY returns byte-identical vectors to sending no
# config at all (it is the server-side default) while RETRIEVAL_DOCUMENT
# differs substantially (cosine 0.83-0.88 against the query-mode vector on
# benchmark-like text), so the parameter is live and only the DOCUMENT side
# actually changes. On gemini-embedding-2 every task_type - all eight valid
# values - returns a byte-identical vector, while an invalid value still
# 400s: the parameter is accepted and validated but has no effect on the
# embedding. Sending it there would buy nothing and would split one embed
# call into two (see EmbeddingProcessor._get_scores_per_side), so -2 is
# deliberately left with no envelope. The probe's output is archived at
# results/diagnostics/protocol/gemini.json; re-probe against the live API
# before assuming this is still true. text-embedding-3-* have no mechanism of
# any kind.
API_MODEL_SCORES_KWARGS = {
    "gemini-embedding-001": {
        "query_encode_kwargs": {"task_type": "RETRIEVAL_QUERY"},
        "document_encode_kwargs": {"task_type": "RETRIEVAL_DOCUMENT"},
    },
}

# The "-no-tok" keys are each method in its OWN default tokenization, with no
# mathematics-aware preprocessing: BM25 and Jaccard fall back to
# whitespace-delimited tokens, TF-IDF to scikit-learn's regex word tokenizer
# plus its own lowercasing. The plain keys use Approach Zero's pya0 tokenizer
# on every LaTeX block (see tokenization_helper.math_word_tokens).
#
# Both are reported: the paper's Table 1 lexical rows are the math-aware ones,
# and Table 3 carries the "-no-tok" rows beside them. The math tokenizer is
# NOT uniformly better - it gains BM25 (+0.018) and Jaccard (+0.009) but costs
# TF-IDF (-0.018) - which is exactly why both belong in the paper.
LEXICAL_MODEL_BUILDERS = {
    "bm25": BM25Processor,
    "tf-idf": TfidfProcessor,
    "jaccard": JaccardProcessor,
    "approach0": Approach0Processor,
    "bm25-no-tok": lambda: BM25Processor(tokenize_approach0=False),
    "tf-idf-no-tok": lambda: TfidfProcessor(tokenize_approach0=False),
    "jaccard-no-tok": lambda: JaccardProcessor(tokenize_approach0=False),
}

QWEN3_RERANKER_REPOS = {
    "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
    "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}

COLBERT_REPOS = {
    "gte-moderncolbert": "lightonai/GTE-ModernColBERT-v1",
    "reason-moderncolbert": "lightonai/Reason-ModernColBERT",
}


INSTRUCTION_EXCLUDED = {
    "reasonir-8b-vllm": (
        "vLLM cannot express ReasonIR's instruction mechanism - its encode() "
        "masks the instruction positions out of the POOLING mask, not out of "
        "the text, and a stock MEAN pooler has no way to represent that. Use "
        "the reasonir-8b key (ReasonIRProcessor) for p1/p2/p3; this key "
        "exists only to run p0 on vLLM"
    ),
    "bm25": (
        "dropped from the instruction experiment 2026-08-30. BM25 has no "
        "instruction mechanism - the prompt can only be prepended to the "
        "query as more query terms - and unlike its two lexical siblings it "
        "has no pathology that makes the result uninterpretable, so it ran "
        "and scored 0.4165 -> 0.3568/0.3630/0.3881 on statement-full. "
        "Reporting one lexical baseline with arms and two without is a "
        "harness accident, not a distinction, so all three are now excluded "
        "together and the runs were deleted"
    ),
    "bm25-no-tok": ("same reason as bm25"),
    "tf-idf-no-tok": (
        "same reason as tf-idf: a document-fitted vocabulary plus cosine "
        "dilution makes instruction words pure noise"
    ),
    "jaccard-no-tok": (
        "same reason as jaccard: instruction tokens inflate the query "
        "token-set union"
    ),
    "tf-idf": (
        "its vocabulary is fitted on documents only and cosine scoring "
        "dilutes real query terms, so instruction words act as pure noise"
    ),
    "jaccard": (
        "instruction tokens inflate the query token-set union, distorting "
        "every score monotonically"
    ),
    "approach0": (
        "its _BROKEN_QUERIES md5 skip-list matches raw query text, so any "
        "query rewrite reintroduces known segfaults"
    ),
}

# Every evaluable key, and what `--models` defaults to.
ALL_MODEL_KEYS = (
    PIPELINE_MODEL_KEYS
    + list(TABLE_MODELS)
    + list(API_MODELS)
    + list(LEXICAL_MODEL_BUILDERS)
)




# ---------------------------------------------------------------------------
# Lookups. Pure functions over the tables above - no argv, no filesystem - so
# the report generators can ask the same questions the runner does.
# ---------------------------------------------------------------------------


def model_spec(model_key: str) -> dict | None:
    """The (HF name, use_vllm, init/scores kwargs) spec for a generically
    resolvable model, or None if this key needs a custom builder."""
    if model_key in GENERIC_MODELS:
        return GENERIC_MODELS[model_key]
    if model_key in TABLE_MODELS:
        return TABLE_MODELS[model_key]
    return None


def model_scores_kwargs(model_key: str) -> dict:
    """Every per-model get_scores kwarg: the preprocessing protocol
    (chunk_to_context/context_length) plus the vendor input envelope
    (query_prompt/document_prompt/suffixes/per-side API params)."""
    if model_key in CUSTOM_MODEL_SCORES_KWARGS:
        return dict(CUSTOM_MODEL_SCORES_KWARGS[model_key])
    if model_key in API_MODEL_SCORES_KWARGS:
        return dict(API_MODEL_SCORES_KWARGS[model_key])
    spec = model_spec(model_key)
    return dict(spec.get("scores_kwargs", {})) if spec is not None else {}


def prompt_scores_kwargs(
    model_key: str, instruction: str | None = None
) -> tuple[dict, bool]:
    """(scores_kwargs, wrap_instruction) for one prompt key.

    wrap_instruction is False when the model's own envelope already carries
    the instruction, so the generic Instruct:/Query: wrap must not be applied
    on top of it (ReasonIR)."""
    kwargs = model_scores_kwargs(model_key)
    wrap_instruction = True

    if model_key == "reasonir-8b" and instruction is not None:
        # The official mechanism, exactly: prepend
        # "<|user|>\n{task}\n<|embed|>\n" to the query and then exclude
        # precisely those tokens from the mean pool. ReasonIRProcessor.encode
        # takes that whole prefix as its `instruction` argument and does the
        # masking itself, so it goes through the per-side encode kwargs
        # rather than as a text affix - and the generic Instruct:/Query: wrap
        # must not be applied on top of it.
        wrap_instruction = False
        kwargs["query_encode_kwargs"] = {
            "instruction": REASONIR_QUERY_INSTRUCTION_SLOT.format(
                instruction=instruction
            )
        }
        kwargs["document_encode_kwargs"] = {"instruction": ""}

    if (
        model_key == "reason-rewriter-reason-embed-8b-instructed"
        and instruction is not None
    ):
        # wrap_instruction stays True: the wrap around the query text is what
        # instructs the REWRITER, and it is also the rewrite cache key, so
        # both composed arms hit the same cached rewrites. This kwarg is the
        # ENCODER half - get_scores reapplies the identical wrap to each
        # rewrite before embedding. Deliberately NOT an AFFIX_KEYS member:
        # those are applied by EmbeddingProcessor, and the envelope assertion
        # in the runner would rightly reject one here, since this processor is
        # a composed ModelProcessor.
        kwargs["embed_instruction"] = instruction

    return kwargs, wrap_instruction


def processor_slot(model_key: str, instruction_key: str) -> str:
    """Which built processor a (model, prompt) cell may reuse. Qwen3-Reranker
    bakes the instruction into the processor, so each prompt needs its own;
    everything else can share one across all prompt keys of a run."""
    if model_key in QWEN3_RERANKER_REPOS:
        return f"qwen3__{instruction_key}"
    return "default"


def uses_tensor_parallel(model_key: str) -> bool:
    return model_key in GENERIC_MODELS_USE_TP or model_key in CUSTOM_MODELS_USE_TP


def is_control_model(model_key: str) -> bool:
    """True when this model has no vendor-documented free-text instruction
    mechanism, so its p1-p3 rows are controls rather than measurements."""
    return model_key in INSTRUCTION_CONTROL_MODELS


def build_spec_processor(spec: dict, model_key: str, tensor_parallel_size: int):
    spec = dict(spec)
    model_name = spec.pop("model")
    init_kwargs = dict(spec.pop("init_kwargs", {}))
    use_vllm = spec.pop("use_vllm", False)
    disable_st_default_prompt = spec.pop("disable_st_default_prompt", False)
    if not use_vllm:
        processor = STProcessor.from_huggingface(model_name, **init_kwargs)
        if disable_st_default_prompt:
            # This model's prompts are applied explicitly per side via
            # scores_kwargs; leaving default_prompt_name set would make
            # sentence-transformers prepend its own default on top.
            st = getattr(processor, "_model", None)
            current = getattr(st, "default_prompt_name", None)
            if current is not None:
                print(
                    f"[~] {model_key}: clearing sentence-transformers "
                    f"default_prompt_name={current!r} (prompts applied per side)"
                )
                st.default_prompt_name = None
        return processor
    if model_key in GENERIC_MODELS_USE_TP and tensor_parallel_size > 1:
        init_kwargs["tensor_parallel_size"] = tensor_parallel_size
    return VLLMProcessor.from_huggingface(model_name, **init_kwargs)


def build_processor(
    model_key: str,
    instruction_key: str,
    tensor_parallel_size: int = 1,
    save_dir: str | Path = "results/evaluation",
):
    """The processor for one (model, prompt) cell.

    Uniform across prompt arms on purpose: a per-arm budget would make p0 and
    p1-p3 differ in token allowance as well as in prompt text, which is not
    the thing being measured."""
    instruction = INSTRUCTIONS[instruction_key]

    if model_key in QWEN3_RERANKER_REPOS:
        # This family has a genuine <Instruct> slot, so p0 must fill it with
        # the VENDOR default to be a real "no task instruction" baseline; the
        # repo's production math instruction lives on as prompt key "pm".
        task_instruction = instruction or VENDOR_QWEN3_RERANKER_INSTRUCTION
        return Qwen3RerankerVLLMProcessor(
            QWEN3_RERANKER_REPOS[model_key], task_instruction=task_instruction
        )
    if model_key == "diver-grouprank-32b":
        return GroupRankProcessor(
            tensor_parallel_size=max(1, tensor_parallel_size),
            scaffold_reserve_tokens=EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE,
        )
    if model_key in COLBERT_REPOS:
        # ColBERT pads queries to query_length with mask tokens (query
        # augmentation), so an arm-dependent query_length would change the
        # query representation even for identical text.
        return ColBERTProcessor(
            COLBERT_REPOS[model_key], query_length=EXPERIMENT_COLBERT_QUERY_LENGTH
        )
    if model_key == "inf-x-retriever":
        return INFXRetrieverProcessor(
            rewrite_log_path=str(
                Path(save_dir) / ".rewrites" / "inf-x-retriever.json"
            )
        )
    if model_key in CUSTOM_MODEL_BUILDERS:
        return CUSTOM_MODEL_BUILDERS[model_key](tensor_parallel_size)
    spec = model_spec(model_key)
    if spec is not None:
        return build_spec_processor(spec, model_key, tensor_parallel_size)
    if model_key in API_MODELS:
        kind, model_name = API_MODELS[model_key]
        if kind == "google":
            return GoogleProcessor(model_name)
        if kind == "openrouter":
            return OpenRouterEmbeddingProcessor(model_name)
        return OpenAIProcessor(model_name)
    if model_key in LEXICAL_MODEL_BUILDERS:
        return LEXICAL_MODEL_BUILDERS[model_key]()
    raise KeyError(f"Unknown model key: {model_key}")
