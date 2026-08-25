"""INF-X-Retriever: infly's two-stage "Query Aligner + Dense Retriever"
system (rank 1 on BRIGHT as of Dec 2025) - inf-query-aligner rewrites the
query, inf-retriever-v1-pro embeds the rewrite (with the instruct prefix)
against prefix-less documents.

Every step below mirrors the team's own released pipeline, verified
line-by-line against three sources on 2026-08-21: the official code at
https://github.com/yaoyichen/INF-X-Retriever (rewrite_queries.py,
retrievers.py, configs/prompts.json), the inf-query-aligner model card's
usage snippet (identical to rewrite_queries.py), and the checkpoints'
generation_config.json / config_sentence_transformers.json. Where our
implementation deviates, the deviation is called out inline.

The official pipeline, exactly:
  1. rewrite_queries.py: chat-template generation with the aligner
     (system = the stock Qwen assistant prompt; user = QUERY_WRITER_PROMPT
     + the query in an "**Input Query:** / **Your Output:**" scaffold),
     tokenized with truncation at 8192, model.generate(max_new_tokens=512)
     - all other sampling knobs come from the checkpoint's
     generation_config.json (do_sample=True, temperature=0.7, top_p=0.8,
     top_k=20, repetition_penalty=1.05). The decoded text REPLACES the
     original query ('line_rewrite["query"] = query_rewrite').
  2. retrievers.py: queries = [instruction + t for t in texts] with the
     single generic instruction from configs/prompts.json ("Instruct: Given
     a web search query, retrieve relevant passages that answer the
     query\\nQuery: " - the SAME string the retriever repo ships as its ST
     "query" prompt; there are NO domain-specific instructions in the
     released configs). Documents get no prefix. inf-retriever-v1-pro via
     AutoModel(trust_remote_code), last_token_pool, L2 normalize, dot
     product.
"""

import hashlib

import numpy as np

from .base import ModelProcessor
from .st_processor import SentenceTransformersProcessor


ALIGNER_REPO = "infly/inf-query-aligner"
RETRIEVER_REPO = "infly/inf-retriever-v1-pro"

# Verbatim from rewrite_queries.py / the aligner's model card. The missing
# space in "instructions.The response" is REAL - the official constant is
# three implicitly-concatenated Python string literals and the second one
# ends without trailing whitespace. Reproduced faithfully, not "fixed":
# the aligner was RL-trained against prompts built from this exact string.
QUERY_WRITER_PROMPT = (
    "For the input query, formulating a concise search query for dense retrieval "
    "by distilling the core intent from a complex user prompt and ignoring LLM instructions."
    "The response should be less than 200 words"
)
ALIGNER_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
ALIGNER_INPUT_MAX_LENGTH = 8192  # official tokenizer(truncation, max_length=8192)
ALIGNER_MAX_NEW_TOKENS = 512  # official model.generate(max_new_tokens=512)

# configs/prompts.json's one-and-only instruction (identical under
# "instructions" and "instructions_long"), byte-identical to the "query"
# prompt in inf-retriever-v1-pro's own config_sentence_transformers.json.
RETRIEVER_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)


def build_inf_retriever_st(model_name: str = RETRIEVER_REPO):
    """The validated inf-retriever-v1-pro SentenceTransformers load, shared
    by the standalone benchmark entry (run_rerankers.py's
    _build_inf_retriever_processor, where the full rationale lives) and the
    INF-X composition below. Short version of what's load-bearing:
    auto-config load (the repo ships Transformer + lasttoken Pooling +
    Normalize), trust_remote_code (bidirectional modeling_qwen.py AND the
    EOS-appending tokenization_qwen.py), torch_dtype auto -> fp16 (the
    checkpoint's declared dtype; ST default would compute fp32), and the
    "message"-modality strip (this tokenizer ships a chat template - the
    ST>=5.7 silent chat-wrapping latch that corrupted the RaDeR family)."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "auto"},
    )
    transformer = st[0]
    if "message" in getattr(transformer, "modality_config", {}):
        transformer.modality_config = {
            k: v for k, v in transformer.modality_config.items() if k != "message"
        }
    return st


class INFXRetrieverProcessor(ModelProcessor):
    """The composed INF-X-Retriever system as one benchmark entry: rewrite
    the query with inf-query-aligner, then delegate scoring of
    (instruction + rewritten query) vs the untouched documents to the exact
    same SentenceTransformersProcessor the standalone inf-retriever-v1-pro
    entry uses - so the standalone-vs-INF-X comparison isolates the query
    rewrite + instruct prefix, with the embedding backend held fixed.

    Deviations from the official pipeline, all deliberate (found by a full
    audit against a clone of the official repo on 2026-08-21 - constants
    were verified BYTE-IDENTICAL programmatically: writer prompt, system
    prompt, user scaffold, instruct prefix, 8192/512 limits):
    - DETERMINISTIC SAMPLING: the official code seeds nothing, so its
      rewrites (do_sample=True per generation_config) are irreproducible
      run-to-run. We keep the official sampling parameters but seed each
      generation from a hash of the query text, inside a fork_rng scope: a
      given query always yields the same rewrite, independent of query
      order - which checkpoint/resume reruns would otherwise change - while
      still sampling from the distribution the aligner was tuned for
      (greedy would NOT be faithful: temperature/top_p/top_k/
      repetition_penalty ship in the checkpoint's own generation_config).
      Relatedly, the official rewrite_queries.py generates in left-padded
      batches (default 256); we generate one query per call - with
      sampling, per-item distributions are unaffected.
    - RETRIEVER DTYPE fp16, not the official eval's implicit fp32: their
      retrievers.py calls AutoModel.from_pretrained with NO torch_dtype,
      which in transformers 4.5x upcasts the fp16 checkpoint to fp32
      compute. We keep the checkpoint's DECLARED fp16 (torch_dtype="auto")
      - required by the 2026-08-21 precision-fairness policy (no model
      computes in fp32), validated harmless for plain encoders (e5
      fp16-vs-fp32: Spearman 1.0000), and it keeps this row's embedding
      backend byte-identical to the standalone inf-retriever-v1-pro row.
    - TRUNCATION AT 32768 not 8192 on the retriever side: the official
      eval truncates query and document encodes at 8192 (doc_max_length
      governs BOTH sides there - its query_max_length arg is dead code);
      our ST load keeps the repo's native max_seq_length 32768. Immaterial
      for this benchmark (only 0.32% of documents even exceed 2048 tokens)
      and the ALIGNER input truncation at 8192 is kept exactly official.

    NOT a deviation: the decoded rewrite is used as-is, no strip - the
    official pipeline's released rewrite_data/*.json confirms every rewrite
    keeps its leading space from decoding, and their retrieval embeds
    instruction + that raw text just like we do.

    Rewrites are cached by raw query text: within one run_rerankers.py
    process the statement-query set is scored under two tasks
    (statement-statement + statement-full), and the rewrite must be
    generated once and reused - both for cost and so the two tasks see the
    same rewrite. When rewrite_log_path is set the cache is also persisted
    as JSON ({original: rewritten}, plus a recipe fingerprint) - primarily
    for AUDITABILITY (the first question a lower-than-standalone score
    raises is "what did the aligner actually write?" - the official repo
    ships its rewrite_data/ for exactly this reason), and secondarily as a
    warm-start for resumed runs (safe: per-query seeding makes regeneration
    byte-identical anyway; a fingerprint mismatch - different prompt/
    aligner/seed scheme - discards the file rather than silently reusing
    stale rewrites). get_scores' check_cache/update_cache kwargs govern the
    rewrite cache exactly like the inner vector cache, so the timing
    harness's check_cache=False measures true per-query rewrite cost.
    """

    processor = "inf-x-retriever"

    # Bump when the rewrite recipe changes (prompt, scaffold, seeding
    # scheme, generation limits) - invalidates persisted rewrite logs.
    _REWRITE_RECIPE_FINGERPRINT = "v1:sha256-seed:fork_rng:8192:512"

    def __init__(
        self,
        aligner_name: str = ALIGNER_REPO,
        retriever_name: str = RETRIEVER_REPO,
        rewrite_log_path: str | None = None,
    ):
        import torch  # noqa: F401 - fail here, not mid-run, if absent
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._aligner_name = aligner_name
        self._retriever_name = retriever_name
        self._rewrite_log_path = rewrite_log_path

        # Official loading kwargs verbatim (torch_dtype="auto" -> bf16, the
        # aligner checkpoint's declared dtype; standard Qwen2ForCausalLM,
        # no remote code).
        self._aligner_tokenizer = AutoTokenizer.from_pretrained(aligner_name)
        self._aligner = AutoModelForCausalLM.from_pretrained(
            aligner_name, torch_dtype="auto", device_map="auto"
        ).eval()

        self._retriever = SentenceTransformersProcessor(
            build_inf_retriever_st(retriever_name), retriever_name
        )

        self._rewrite_cache: dict[str, str] = self._load_rewrite_log()

    def _load_rewrite_log(self) -> dict[str, str]:
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
        if data.get("fingerprint") != self._REWRITE_RECIPE_FINGERPRINT or data.get(
            "aligner"
        ) != self._aligner_name:
            # Stale recipe - regenerating is always safe (and correct).
            return {}
        rewrites = data.get("rewrites", {})
        return dict(rewrites) if isinstance(rewrites, dict) else {}

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
                        "aligner": self._aligner_name,
                        "rewrites": self._rewrite_cache,
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            tmp.replace(p)
        except OSError as e:
            # Auditability must never kill a scoring run.
            print(f"[!!] inf-x-retriever: rewrite log write failed ({e})")

    @property
    def model(self) -> str:
        return f"{self._aligner_name} + {self._retriever_name}"

    def _rewrite(
        self, query: str, *, check_cache: bool = True, update_cache: bool = True
    ) -> str:
        import torch

        if check_cache and query in self._rewrite_cache:
            return self._rewrite_cache[query]

        messages = [
            {"role": "system", "content": ALIGNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{QUERY_WRITER_PROMPT}\n\n"
                    f"**Input Query:**\n{query}\n"
                    f"**Your Output:**\n"
                ),
            },
        ]
        text = self._aligner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._aligner_tokenizer(
            [text],
            truncation=True,
            max_length=ALIGNER_INPUT_MAX_LENGTH,
            return_tensors="pt",
        ).to(self._aligner.device)

        # Content-derived seed: stable across runs, processes, and query
        # order. fork_rng keeps the seeding from perturbing any other RNG
        # consumer in the process.
        seed = int.from_bytes(
            hashlib.sha256(query.encode("utf-8")).digest()[:4], "big"
        )
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                generated = self._aligner.generate(
                    **inputs, max_new_tokens=ALIGNER_MAX_NEW_TOKENS
                )

        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        rewritten = self._aligner_tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )
        # Guard OUR pipeline (the official one never hits this: BRIGHT
        # rewrites are non-empty): an empty rewrite would zero-norm-crash
        # the cosine downstream, so fall back to the original query rather
        # than dying 900 queries into a run.
        if not rewritten.strip():
            rewritten = query

        if update_cache:
            self._rewrite_cache[query] = rewritten
            self._save_rewrite_log()
        return rewritten

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float]:
        # check_cache/update_cache govern the rewrite cache exactly like the
        # inner vector cache; peeked (not popped) because the inner
        # get_scores consumes the same flags.
        rewritten = self._rewrite(
            query,
            check_cache=kwargs.get("check_cache", True),
            update_cache=kwargs.get("update_cache", True),
        )
        instructed = RETRIEVER_QUERY_INSTRUCTION + rewritten
        # Full delegation: the inner processor's vector cache (documents are
        # re-scored across queries and tasks), batch_size slicing, and
        # cosine scoring all behave exactly as in the standalone entry.
        # Cosine over the ST stack's L2-normalized outputs == the official
        # normalized dot product (the *100 scaling there is rank-neutral).
        return self._retriever.get_scores(
            query=instructed,
            documents=documents,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

    def export_cache(self, path: str) -> None:
        self._retriever.export_cache(path)

    def import_cache(self, path: str) -> None:
        self._retriever.import_cache(path)

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Raw embedding access (no rewrite, no instruction) - the
        retriever backend as-is, for parity checks/diagnostics."""
        return self._retriever.encode(texts, **kwargs)
