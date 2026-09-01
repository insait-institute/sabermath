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
        return self._retriever.encode(texts, **kwargs)
