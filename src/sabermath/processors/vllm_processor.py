"""vLLM-backed dense embedding processor.

Beyond the plain llm.embed() wrapper this originally was, VLLMProcessor now
carries the pieces needed to serve models whose Hugging Face repos aren't
directly loadable/configurable by vLLM - all validated model-by-model against
the previous production paths in scripts/test_vllm_feasibility.py
(results/vllm_feasibility/summary.json, 2026-08-19):

- pooler_config as a PLAIN DICT (version-tolerant field mapping, see
  make_pooler_config) so callers like run_rerankers.py's GENERIC_MODELS can
  express pooling overrides as JSON-able init_kwargs;
- export_clean: a one-off HF AutoModel re-export for checkpoints vLLM's
  loader can't consume directly (bert-base-uncased's MLM checkpoint prefixes
  every tensor with 'bert.', which vLLM's BertModel loader doesn't strip);
- client-side chunk_to_context (chunk+mean, mirroring
  SentenceTransformersProcessor.encode's semantics exactly) and silent
  truncation-to-context, because vLLM hard-REJECTS overlong prompts where
  sentence-transformers silently truncates;
- client-side batch_size slicing of embed calls - vLLM has no such knob
  (it schedules/micro-batches internally); the timing harness
  (scripts/measure_query_time.py) uses this to standardize request-level
  parallelism at 16 documents in flight across every model.
"""

import os
from pathlib import Path

import numpy as np

from .embedding_processor import EmbeddingProcessor


def _get_model_name(llm) -> str | None:
    return getattr(
        getattr(getattr(llm, "llm_engine", None), "model_config", None), "model", None
    ) or getattr(getattr(llm, "model_config", None), "model", None)


def artifact_cache_dir() -> Path:
    """Where one-off derived checkpoints (clean re-exports, merged LoRA
    adapters) are cached. Deliberately NOT under results/ - that directory
    rides the region rsync sync-back (scripts/rerankers/_common.sh) and a
    ~16GB merged model must never be pushed over SSH on every job exit.
    /scratch is per-node, so a first job on a fresh node pays the derivation
    once; that's accepted."""
    env = os.environ.get("SABERMATH_VLLM_EXPORT_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch") / os.environ.get("USER", "nouser")
    if scratch.is_dir():
        return scratch / "sabermath_vllm_artifacts"
    return Path.home() / ".cache" / "sabermath" / "vllm_artifacts"


def make_pooler_config(fields: dict):
    """Build a vllm PoolerConfig from a plain dict, tolerating field renames
    across vLLM versions. Callers use the canonical names pooling_type /
    normalize / activation; vllm==0.26.0 (the pinned version) has neither
    `normalize` nor `activation` - its single `use_activation` field does
    both jobs (for the embed pooler the "activation" IS the final L2
    normalize; for classify it's the sigmoid/softmax). Unknown fields are
    dropped with a warning so a rename never turns into a hard failure."""
    import inspect

    from vllm.config import PoolerConfig

    try:
        accepted = set(inspect.signature(PoolerConfig).parameters)
    except (TypeError, ValueError):
        accepted = None  # couldn't introspect - just try everything

    aliases = {
        "normalize": ["use_activation"],
        "activation": ["use_activation", "softmax"],
        "softmax": ["use_activation", "activation"],
    }
    out, dropped = {}, []
    for key, value in fields.items():
        if accepted is None or key in accepted:
            out[key] = value
            continue
        for alt in aliases.get(key, []):
            if alt in accepted:
                out[alt] = value
                break
        else:
            dropped.append(key)
    if dropped:
        print(f"[~] PoolerConfig on this vLLM build has no {dropped} - dropped.")
    return PoolerConfig(**out)


def export_clean_checkpoint(model_name: str, cache_dir: str | Path | None = None) -> str:
    """One-off re-export of a checkpoint through HF AutoModel: strips
    task-head weight prefixes (e.g. bert-base-uncased's MLM checkpoint names
    everything 'bert.*'), drops the task head, and writes a config whose
    architectures matches the bare encoder class vLLM can load. Cached -
    subsequent calls on the same machine reuse the export."""
    base = Path(cache_dir) if cache_dir is not None else artifact_cache_dir()
    out = base / (model_name.replace("/", "__") + "-clean")
    if (out / "config.json").exists():
        print(f"[~] Reusing already-exported clean checkpoint at {out}")
        return str(out)

    from transformers import AutoModel, AutoTokenizer

    print(f"[~] Re-exporting {model_name} through AutoModel (one-off, cached)...")
    out.mkdir(parents=True, exist_ok=True)
    AutoModel.from_pretrained(model_name).save_pretrained(out)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(out)
    return str(out)


def _chunk_texts(texts, tokenizer, max_len):
    """Client-side replica of SentenceTransformersProcessor.encode's
    chunk_to_context: split each text into max_len-sized token chunks (minus
    special tokens); the caller mean-averages per owner afterwards. Keep in
    sync with st_processor.py."""
    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    chunk_token_len = max_len - num_special
    if chunk_token_len <= 0:
        raise ValueError(f"Context length {max_len} too small for special tokens.")

    def _decode(ids):
        # skip_special_tokens=False, deliberately. The encode above passes
        # add_special_tokens=False, so nothing here was added by us - the only
        # special tokens present are ones the CALLER put in the text, and a
        # caller that asks for them means them. Dropping them silently deleted
        # the RaDeR bi-encoders' "<|im_end|>" suffix from every text (verified
        # 2026-08-25: even a short single-chunk document came back without it),
        # which for a LAST-token pooler is exactly the position being pooled.
        return tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _fit(part):
        # The decode->re-encode roundtrip is NOT length-stable (confirmed on
        # multilingual-e5-large: a 510-token slice re-encoded to 511
        # sentencepiece tokens, and vLLM hard-rejects >max_model_len prompts
        # instead of truncating like ST does). Re-truncate until the chunk
        # actually fits when vLLM re-tokenizes it.
        for _ in range(5):
            ids = tokenizer.encode(part, add_special_tokens=False, truncation=False)
            if len(ids) <= chunk_token_len:
                return part
            part = _decode(ids[:chunk_token_len])
        return _decode(ids[: max(1, 2 * chunk_token_len - len(ids))])

    all_chunks, owners = [], []
    for text_idx, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not token_ids:
            all_chunks.append("")
            owners.append(text_idx)
            continue
        for start in range(0, len(token_ids), chunk_token_len):
            all_chunks.append(_fit(_decode(token_ids[start : start + chunk_token_len])))
            owners.append(text_idx)
    return all_chunks, owners


def _truncate_text(text, tokenizer, max_len):
    """Silent truncation-to-context, mirroring what sentence-transformers
    does implicitly (vLLM instead raises on overlong prompts). Includes the
    same roundtrip-stability loop as _chunk_texts."""
    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_len - num_special
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) <= budget:
        return text
    # skip_special_tokens=False for the same reason as in _chunk_texts.
    for _ in range(5):
        text = tokenizer.decode(
            ids[:budget], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(ids) <= budget:
            return text
    return tokenizer.decode(
        ids[: max(1, 2 * budget - len(ids))],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


class VLLMProcessor(EmbeddingProcessor):
    processor = "vllm"

    def __init__(
        self,
        llm: "vllm.LLM",
        model_name: str | None = None,
        tokenizer_name: str | None = None,
    ):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("Please install vllm to use it as a processor") from e

        if not isinstance(llm, LLM):
            raise TypeError(
                "VLLMProcessor can only be build using a valid vllm.LLM object"
            )

        self._model_name = model_name or _get_model_name(llm)
        # The repo the tokenizer loads from may differ from the display name
        # (export_clean loads from the re-exported local dir).
        self._tokenizer_name = tokenizer_name or self._model_name
        self._tokenizer = None
        self._llm = llm

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        *,
        pooler_config: dict | None = None,
        export_clean: bool = False,
        **kwargs,
    ) -> "VLLMProcessor":
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("Please install vllm to use it as a processor") from e

        load_name = (
            export_clean_checkpoint(model_name) if export_clean else model_name
        )

        # Long-standing defaults, now overridable via kwargs instead of
        # colliding with them (dtype / hf_overrides / max_model_len /
        # tensor_parallel_size always passed straight through).
        llm_kwargs = dict(
            runner="pooling",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
        )
        llm_kwargs.update(kwargs)

        if pooler_config is not None:
            config = (
                make_pooler_config(pooler_config)
                if isinstance(pooler_config, dict)
                else pooler_config
            )
            # vllm==0.26.0 spells it pooler_config; older builds used
            # override_pooler_config - try both rather than hard-failing on
            # a rename.
            last_error = None
            for kwarg_name in ("pooler_config", "override_pooler_config"):
                try:
                    llm = LLM(model=load_name, **{kwarg_name: config}, **llm_kwargs)
                    break
                except TypeError as e:
                    last_error = e
            else:
                raise last_error
        else:
            llm = LLM(model=load_name, **llm_kwargs)

        return cls(llm, model_name, tokenizer_name=load_name)

    @property
    def model(self) -> str | None:
        return self._model_name

    def _get_tokenizer(self):
        """HF tokenizer for client-side chunking/truncation. None when this
        processor was built from a raw vllm.LLM with no resolvable name -
        encode() then falls back to passing texts through untouched (the
        pre-extension behavior)."""
        if self._tokenizer is None and self._tokenizer_name:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name, trust_remote_code=True
            )
        return self._tokenizer

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        *,
        batch_size: int | None = None,
        chunk_to_context: bool = False,
        context_length: int | None = None,
        truncate_to: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Embed texts. Output is STRICTLY positional (texts[i] -> row i, no
        deduplication) - EmbeddingProcessor's per-text cache and the timing
        harness's check_cache=False path both rely on that.

        batch_size: client-side slicing of llm.embed calls (vLLM has no such
        knob itself); None = one call over everything, the production default.
        chunk_to_context/context_length: chunk+mean, same semantics as
        SentenceTransformersProcessor.encode.
        truncate_to: explicit truncation limit; when neither chunking nor an
        explicit limit is given, texts are silently truncated to the
        tokenizer's model_max_length (if it's a real limit) - mirroring ST,
        since vLLM rejects overlong prompts outright."""
        if not texts:
            return np.empty((0, 0), dtype=float)

        tokenizer = self._get_tokenizer() if self._tokenizer_name else None

        owners = None
        if chunk_to_context:
            if tokenizer is None:
                raise ValueError(
                    "chunk_to_context requires a resolvable tokenizer name"
                )
            max_len = context_length or getattr(tokenizer, "model_max_length", None)
            if max_len is None or max_len >= 1_000_000:
                raise ValueError("Could not determine model context length")
            prompts, owners = _chunk_texts(texts, tokenizer, max_len)
        else:
            limit = truncate_to
            if limit is None and tokenizer is not None:
                mml = getattr(tokenizer, "model_max_length", None)
                if mml is not None and mml < 1_000_000:
                    limit = mml
            if limit is not None and tokenizer is not None:
                prompts = [_truncate_text(t, tokenizer, limit) for t in texts]
            else:
                prompts = list(texts)

        embeddings: list = []
        step = batch_size if batch_size and batch_size > 0 else len(prompts)
        for start in range(0, len(prompts), step):
            outputs = self._llm.embed(
                prompts[start : start + step],
                use_tqdm=show_progress_bar,
                **kwargs,
            )
            embeddings.extend(o.outputs.embedding for o in outputs)

        vectors = np.asarray(embeddings, dtype=np.float64)
        if owners is None:
            return vectors

        owners_arr = np.asarray(owners)
        out = np.empty((len(texts), vectors.shape[1]), dtype=np.float64)
        for i in range(len(texts)):
            out[i] = vectors[owners_arr == i].mean(axis=0)
        return out
