import os
import sys
from pathlib import Path

# Set CUDA devices strictly if needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# math-vs-word has its own environment.yml, separate from the main
# sabermath package's - so `import sabermath` isn't guaranteed to resolve
# without this, same convention as scripts/measure_query_time.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
# Every model here is built and scored through scripts/run_rerankers.py's
# own functions rather than a second set of recipes - see the block below
# for what that fixed. Importable safely: everything used from it lives
# below its `if __name__ == "__main__":` guard, so there are no
# argparse/model-loading side effects at import time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from run_rerankers import RADER_BIENCODER_MODELS  # noqa: E402
from sabermath.processors import SentenceTransformersProcessor  # noqa: E402

# HOW MODELS ARE BUILT AND CALLED HERE (changed 2026-08-26)
# --------------------------------------------------------
# This file no longer maintains its own roster of builders and per-model
# get_scores kwargs. Every model is built and scored by delegating to
# scripts/run_rerankers.py's own functions:
#
#     rr._build_experiment_processor(key, instruction_key, tp, save_dir, legacy)
#     rr._experiment_scores_kwargs(key, instruction_text, legacy)
#
# which is exactly what scripts/run_dedup.py does (run_dedup.py:188-199 and
# 291-295). The main experiment, the dedup experiment and math-vs-word now go
# through one code path, so "how do we call this model" has exactly one
# answer per model, defined in one place.
#
# WHY: the parallel roster had drifted badly from run_rerankers.py. It was
# written when the only models here were the original 17 and the only overlap
# with run_rerankers was qwen3-embedding-8b in GENERIC_MODELS - the claim the
# old comment here made ("No other model in ALLOWED_MODELS appears in
# GENERIC_MODELS at all"). run_rerankers.TABLE_MODELS then grew to cover 16 of
# those 17, and this file was never updated, so as of 2026-08-26 it differed
# from the paper's own harness on:
#
#   * BACKEND, 15 models. TABLE_MODELS runs qwen3-embedding-4b/0.6b, bge-m3,
#     kalm-12b, embeddinggemma-300m, bert-base-uncased, roberta-base, all
#     three harrier, octen-4b/8b, llama-embed-nemotron-8b, e5-large and
#     jina-v5-small under vLLM (the 2026-08-20 rollout, each FEASIBLE-verified
#     at Spearman 0.999-1.0). This file ran every one of them through
#     SentenceTransformers instead.
#   * INPUT ENVELOPE, 10 models. Dropped entirely: the RaDeR family's
#     "query: "/"document: " markers plus RADER_EXPECTED_EOS suffixes,
#     EmbeddingGemma's "task: search result | query: "/"title: none | text: ",
#     both jina-v5 "Query: "/"Document: ", and gemini-embedding-001's
#     RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT task_type.
#   * CHUNKING, 3 models. bert-base-uncased, roberta-base and e5-large take
#     _CHUNK512_SCORES_KWARGS in the main experiment; here they silently
#     truncated at the ST default instead.
#   * PROCESSOR ARGUMENTS, 6 models. Both ColBERTs ran at the checkpoint's own
#     query_length instead of EXPERIMENT_COLBERT_QUERY_LENGTH - and ColBERT
#     pads queries to that length with mask tokens, so it changes the query
#     representation for identical text. diver-grouprank-32b ran without
#     EXPERIMENT_GROUPRANK_SCAFFOLD_RESERVE; all three qwen3-rerankers ran
#     with an empty <Instruct> slot rather than the vendor default.
#   * PROCESSOR CLASS, 2 models. text-embedding-3-large/small went through a
#     local copy of OpenRouterEmbeddingProcessor that was not an
#     EmbeddingProcessor at all - no vector cache, and no way to apply an
#     envelope.
#
# The old ST-crash workarounds for the four Gemma3/Octen models
# (_build_gemma3_text_embedding_processor / _build_octen_processor) are gone
# with the ST path itself: those four are use_vllm=True in TABLE_MODELS.
#
# DO NOT RESURRECT THEM FROM GIT HISTORY. _build_octen_processor was not just
# unnecessary, it was WRONG. Measured 2026-08-26 on 14 probe texts
# (scripts/diag_tokenization.py):
#
#     hand-built stack vs vLLM : cosine 0.704
#     GENERIC ST load  vs vLLM : cosine 0.99978
#
# Its premise - that the generic SentenceTransformer load crashes on Octen's
# 2_Normalize config - is VERSION-DEPENDENT, and false on the version this
# experiment actually runs. Measured directly:
#     sentence-transformers 5.7.0 : generic load SUCCEEDS
#     sentence-transformers 6.0.0 : generic load raises
#         TypeError: Normalize.__init__() got an unexpected keyword
#         argument 'normalize_embeddings'
# So on 6.0.0 the ST path has NO correct option for this model: the generic
# load is unavailable and the hand-built fallback is wrong (0.68 vs vLLM
# there, 0.70 on 5.7.0). That is an additional reason to keep Octen on vLLM
# rather than to reinstate an ST builder. The hand-built imitation of the vendor stack
# produced embeddings ~0.3 cosine away from both the generic stack and vLLM,
# which is why the pre-delegation Octen numbers moved +6.60pp when this file
# started delegating. vLLM is the correct backend for Octen; the old ST
# numbers were the broken ones. See the octen-embedding-* entry in
# scripts/run_rerankers.py for the full elimination trail.
#
# CONSEQUENCE FOR EXISTING RESULTS: every file in similarities/ predates this
# change and was produced by the old path. The default arm here is now the
# CANONICAL protocol (legacy=False), the same one run_rerankers.py and
# run_dedup.py default to; pass --legacy to calc_sims.py to reproduce the old
# prompt-free numbers, which writes to <method>__legacy.json so the two can
# never overwrite each other. See scripts/instructions/PROTOCOL.md.
#
# THE ONE EXCEPTION: microsoft/codebert-base is not in the paper's model
# table and has no run_rerankers.py key, so there is no canonical call for it
# to match. It keeps the generic SentenceTransformers path, and is the only
# model that does.
_NO_EXPERIMENT_KEY = {"microsoft/codebert-base"}

# tensor_parallel_size. This module pins CUDA_VISIBLE_DEVICES=0 above, so
# there is exactly one visible GPU and tp>1 is not expressible. Matches
# submit_sims.sh, which allocates --gpus=h200:1 per job.
_TENSOR_PARALLEL_SIZE = 1

# Only ever read by run_rerankers' inf-x-retriever builder, which is not in
# ALLOWED_MODELS - but _build_experiment_processor takes it unconditionally.
_SAVE_DIR = str(Path(__file__).resolve().parent / "similarities")


# HF repo string -> the short model key scripts/run_rerankers.py uses, built
# by inverting that file's own spec dicts rather than hand-maintaining a
# second roster here. Every call below goes through this: the model key IS
# the thing that decides how the model is built and scored.
def _model_key_by_id() -> dict:
    import run_rerankers as rr

    mapping = {}
    for spec_dict in (rr.GENERIC_MODELS, getattr(rr, "TABLE_MODELS", {})):
        for key, spec in spec_dict.items():
            if isinstance(spec, dict) and "model" in spec:
                mapping[spec["model"]] = key
    for key, repo in RADER_BIENCODER_MODELS.items():
        mapping[repo] = key
    for attr in ("QWEN3_RERANKER_REPOS", "COLBERT_REPOS"):
        for key, repo in getattr(rr, attr, {}).items():
            mapping[repo] = key
    for key, value in getattr(rr, "API_MODELS", {}).items():
        mapping.setdefault(value[1] if isinstance(value, tuple) else value, key)
    return mapping


def model_key_for(model_id: str) -> str:
    """The run_rerankers model key for a math-vs-word method id.

    Methods that ARE already a short key (the RERANK entries, which have no
    single natural HF repo) map to themselves.
    """
    import run_rerankers as rr

    if model_id in rr.EXPERIMENT_MODEL_KEYS:
        return model_id
    key = _model_key_by_id().get(model_id)
    if key is None:
        # Models built from CUSTOM_MODEL_BUILDERS/API_MODELS are keyed by a
        # short name with no "model" field to invert, but that short name is
        # the repo's last path component lowercased for every one of them
        # (reasonir/ReasonIR-8B -> reasonir-8b, jhu-clsp/rank1-32b ->
        # rank1-32b, google/gemini-embedding-001 -> gemini-embedding-001).
        # Only accepted when the result is a real key, so a model genuinely
        # outside the paper's table (microsoft/codebert-base) still raises
        # rather than silently resolving to nothing.
        candidate = model_id.rsplit("/", 1)[-1].lower()
        if candidate in rr.EXPERIMENT_MODEL_KEYS:
            return candidate
    if key is None:
        raise KeyError(
            f"{model_id} has no scripts/run_rerankers.py model key, so its "
            "input envelope and instruction handling are unknown. Add it to "
            "that file's spec dicts rather than special-casing it here."
        )
    return key


def _instruction_args(instruction_key: str | None) -> tuple[str, str | None]:
    """(instruction_key, instruction_text) as run_rerankers wants them.

    instruction_key=None means this experiment's default arm. It maps to
    "p0", whose INSTRUCTIONS entry is None - i.e. "no instruction text", the
    same arm run_dedup.py pins itself to (it passes "p0" to the processor
    builder and None as the instruction text). p0 is NOT prompt-free: under
    the canonical protocol the model's vendor envelope still applies.
    """
    from sabermath.instructions import INSTRUCTIONS

    if instruction_key is None:
        return "p0", INSTRUCTIONS["p0"]
    if instruction_key not in INSTRUCTIONS:
        raise ValueError(
            f"Unknown instruction key {instruction_key!r} - valid: "
            f"{sorted(INSTRUCTIONS)}"
        )
    return instruction_key, INSTRUCTIONS[instruction_key]


def get_scores_kwargs(
    model_id: str,
    instruction_key: str | None = None,
    legacy: bool = False,
) -> dict:
    """Every kwarg sim_embeddings.py must forward to processor.get_scores().

    Delegates to run_rerankers._experiment_scores_kwargs, so this is the
    same preprocessing protocol (chunk_to_context/context_length) and the
    same vendor input envelope (query_prompt/document_prompt/suffixes/
    per-side API params) the main experiment and run_dedup.py use.

    legacy=True strips the envelope only, reproducing the pre-2026-08-25
    prompt-free protocol - identical in meaning to run_rerankers.py's and
    run_dedup.py's own --legacy.
    """
    if model_id in _NO_EXPERIMENT_KEY:
        return {}

    import run_rerankers as rr

    key, instruction_text = _instruction_args(instruction_key)
    kwargs, _ = rr._experiment_scores_kwargs(
        model_key_for(model_id), instruction_text, legacy=legacy
    )
    return kwargs


def wraps_instruction(
    model_id: str,
    instruction_key: str | None = None,
    legacy: bool = False,
) -> bool:
    """Whether the caller should wrap the query with the generic
    "Instruct: ...\\nQuery: ..." template.

    False when the model carries the instruction through its own mechanism
    and the generic wrap must NOT be applied on top:

      * reasonir-8b - its encode() takes the instruction as an argument and
        masks exactly those tokens out of the mean pool, so the instruction
        reaches it via query_encode_kwargs, not as query text.
      * the qwen3-reranker family - the instruction fills the model's own
        <Instruct> slot at construction time (run_rerankers sets
        query_instruction=None for them, run_rerankers.py:1372-1375).

    The old code here discarded run_rerankers' wrap_instruction flag and
    wrapped unconditionally, which double-applied the instruction for
    reasonir-8b and applied it twice over for the qwen3-rerankers.
    """
    if model_id in _NO_EXPERIMENT_KEY:
        return True

    import run_rerankers as rr

    key, instruction_text = _instruction_args(instruction_key)
    model_key = model_key_for(model_id)
    _, wrap = rr._experiment_scores_kwargs(model_key, instruction_text, legacy=legacy)
    if model_key in rr.QWEN3_RERANKER_REPOS:
        return False
    return wrap


def assert_envelope_supported(model_id: str, processor, scores_kwargs: dict) -> None:
    """run_rerankers' own guard: envelope kwargs are applied by
    EmbeddingProcessor.get_scores, so a cross-encoder would silently swallow
    them. Fail loudly instead of scoring with a dropped envelope."""
    if model_id in _NO_EXPERIMENT_KEY:
        return

    import run_rerankers as rr

    rr._assert_envelope_supported(model_key_for(model_id), processor, scores_kwargs)


def get_model(
    MODEL_ID: str,
    instruction_key: str | None = None,
    legacy: bool = False,
):
    """The processor for MODEL_ID, built exactly as the main experiment and
    run_dedup.py build it.

    instruction_key matters at BUILD time, not just at scoring time, for
    three families - the qwen3-rerankers (the instruction fills their
    <Instruct> slot), diver-grouprank-32b (per-arm scaffold token reserve)
    and the ColBERTs (query_length) - which is why it is taken here and not
    only in get_scores_kwargs.
    """
    if MODEL_ID in _NO_EXPERIMENT_KEY:
        print(
            f"Loading {MODEL_ID} via SentenceTransformersProcessor."
            f"from_huggingface (no run_rerankers key - see _NO_EXPERIMENT_KEY)..."
        )
        return SentenceTransformersProcessor.from_huggingface(
            MODEL_ID, trust_remote_code=True
        )

    import run_rerankers as rr

    key, _ = _instruction_args(instruction_key)
    model_key = model_key_for(MODEL_ID)
    protocol = "legacy" if legacy else "canonical"
    print(
        f"Loading {MODEL_ID} as run_rerankers model key {model_key!r} "
        f"(arm {key}, {protocol} protocol) via "
        f"run_rerankers._build_experiment_processor..."
    )
    return rr._build_experiment_processor(
        model_key, key, _TENSOR_PARALLEL_SIZE, _SAVE_DIR, legacy=legacy
    )

ALLOWED_MODELS = [
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "google/embeddinggemma-300m",
    "google-bert/bert-base-uncased",  # Standard BERT
    "FacebookAI/roberta-base",  # Standard RoBERTa
    "microsoft/codebert-base",
    "google/gemini-embedding-001",
    "google/gemini-embedding-2",
    "microsoft/harrier-oss-v1-0.6b",
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-27b",
    "Octen/Octen-Embedding-4B",
    "Octen/Octen-Embedding-8B",
    "jinaai/jina-embeddings-v5-text-nano",
]

# Added 2026-08-22 to cover the rest of the paper's model table beyond the
# 17 above (which were the original math-vs-word roster). How each one is
# built and scored is not decided here - it is whatever run_rerankers.py's
# spec dicts say for that model's key, reached via model_key_for(). The two
# lists differ only in when they were added; they get identical treatment.
ADDITIONAL_MODELS = [
    # EMBED
    "hanhainebula/reason-embed-qwen3-8b-0928",  # Reason-Embed-Qwen3-8B
    "AQ-MedAI/Diver-Retriever-4B",
    "AQ-MedAI/Diver-Retriever-0.6B",
    "infly/inf-retriever-v1-pro",  # INF-Retriever-v1-Pro
    "reasonir/ReasonIR-8B",  # ReasonIR-8B
    RADER_BIENCODER_MODELS["rader-14b"],
    RADER_BIENCODER_MODELS["rader-7b"],
    RADER_BIENCODER_MODELS["rader-3b"],
    "nvidia/llama-embed-nemotron-8b",  # LLaMA-Embed-Nemotron-8B
    "intfloat/multilingual-e5-large",  # Multilingual-E5-Large
    "jinaai/jina-embeddings-v5-text-small",  # Jina-v5-Text-Small
    # API (OpenRouter - run_rerankers.API_MODELS routes these)
    "text-embedding-3-large",
    "text-embedding-3-small",
    # RERANK
    "jhu-clsp/rank1-32b",  # Rank1-32B
    # Rank1 size ablations, added 2026-08-26. Keyed by HF repo the same way
    # rank1-32b is: model_key_for() resolves these through its
    # last-path-component-lowercased fallback ("jhu-clsp/rank1-7b" ->
    # "rank1-7b"), which is a real CUSTOM_MODEL_BUILDERS key.
    "jhu-clsp/rank1-7b",  # Rank1-7B
    "jhu-clsp/rank1-0.5b",  # Rank1-0.5B
    "qwen3-reranker-8b",
    "qwen3-reranker-4b",
    "qwen3-reranker-0.6b",
    "splade-code-8b",
    "splade-code-0.6b",
    "rader-reranker-7b",  # RaDeR-Reranker-7B
    "diver-grouprank-32b",  # Diver-GroupRank-32B
    "lightonai/GTE-ModernColBERT-v1",  # GTE-ModernColBERT
    "lightonai/Reason-ModernColBERT",  # Reason-ModernColBERT
    # INF-X-Retriever, added 2026-08-26. Short-keyed, not HF-repo-keyed: it
    # is a composed SYSTEM (inf-query-aligner rewrites the query, then the
    # rewrite is embedded against prefix-less documents by the retriever
    # half), so there is no single repo to dispatch on - same convention as
    # diver-grouprank-32b / rader-reranker-7b.
    "inf-x-retriever",  # INF-X-Retriever
    # math-vs-word rewritten arm, added 2026-08-28. All four are short-keyed
    # for the same reason inf-x-retriever is: three are composed SYSTEMS with
    # no single repo to dispatch on, and retro-star-32b is a
    # CUSTOM_MODEL_BUILDERS entry whose key is not its repo's last path
    # component. model_key_for() takes all four via its
    # "already a short key" branch.
    #
    # The three -rewritten rows put the instruction on the REWRITER only:
    # run_rerankers wraps it into the query text, which is what _rewrite()
    # consumes, and their query side then embeds/scores the rewrite. That is
    # the default composed behaviour - do NOT substitute
    # reason-rewriter-reason-embed-8b-instructed here, which additionally
    # instructs the encoder half and is a different experiment.
    "retro-star-32b",  # Retro*-Qwen3-32B
    "retro-star-32b-rewritten",  # Retro*-Qwen3-32B on the rewritten query
    "reason-rewriter-reason-embed-8b",  # ReasonEmbed-Qwen3-8B-rewritten
    "reason-rewriter-reason-embed-llama-3.1-8b",  # ReasonEmbed-Llama-3.1-8B-rewritten
    # ReasonEmbed-LLaMA-3.1-8B, added 2026-08-29. HF-repo-keyed like the Qwen3
    # sibling above rather than short-keyed: it is a plain GENERIC_MODELS
    # bi-encoder, so _model_key_by_id() inverts its spec["model"] straight to
    # the "reason-embed-llama-3.1-8b" key. Only the composed rows need a short
    # key. This is the STANDALONE encoder - the rewritten counterpart is the
    # reason-rewriter-... entry directly above, and the pair is what makes the
    # rewrite's effect on the math/word split readable for this checkpoint.
    "hanhainebula/reason-embed-llama-3.1-8b-0928",  # ReasonEmbed-Llama-3.1-8B
]

ALLOWED_MODELS = ALLOWED_MODELS + ADDITIONAL_MODELS
