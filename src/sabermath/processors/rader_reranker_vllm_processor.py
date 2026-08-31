from pathlib import Path
from typing import ClassVar

from .base import ModelProcessor
from .vllm_processor import artifact_cache_dir, make_pooler_config

DEFAULT_LORA_MODEL = "Raderspace/reranker_Qwen25_7B_NuminaMath_MATH_allquerytypes"
DEFAULT_MAX_LENGTH = 8192  # the HF reference implementation's default


class RaDeRRerankerVLLMProcessor(ModelProcessor):
    processor: ClassVar[str | None] = "rader-reranker-vllm"

    def __init__(
        self,
        lora_model_name: str = DEFAULT_LORA_MODEL,
        *,
        base_model_name: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int | None = None,
        merged_dir: str | Path | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        # batch_size: client-side slicing of llm.classify calls (None = one
        # call over all candidates, the production default; timing passes 16).
        self._lora_model_name = lora_model_name
        self._base_model_name = base_model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._merged_dir = Path(merged_dir) if merged_dir is not None else None
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization

        self._llm = None
        self._tokenizer = None

    @property
    def model(self) -> str | None:
        return self._lora_model_name

    def _merge(self) -> Path:
        merged = self._merged_dir
        if merged is None:
            merged = artifact_cache_dir() / (
                self._lora_model_name.replace("/", "__") + "-merged"
            )
        if (merged / "config.json").exists():
            print(f"[~] Reusing already-merged model at {merged}")
            return merged

        import torch

        try:
            from peft import PeftConfig, PeftModel
        except ImportError as e:
            raise ImportError(
                "Please install peft to use RaDeRRerankerVLLMProcessor (the "
                "one-off LoRA merge needs it - see scripts/envs/env_vllm.yml)"
            ) from e
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        base_name = self._base_model_name
        if base_name is None:
            base_name = PeftConfig.from_pretrained(
                self._lora_model_name
            ).base_model_name_or_path

        print(
            f"[~] Merging {self._lora_model_name} into {base_name} "
            f"(one-off, cached at {merged}, ~16GB)..."
        )
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_name, num_labels=1, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, self._lora_model_name)
        model = model.merge_and_unload()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        merged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged)
        tokenizer.save_pretrained(merged)
        print(f"[+] Merged model saved to {merged}")
        return merged

    def _init(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "Please install vllm to use RaDeRRerankerVLLMProcessor"
            ) from e
        from transformers import AutoTokenizer

        merged = self._merge()

        pooler = make_pooler_config({"pooling_type": "LAST", "activation": False})
        base = dict(
            model=str(merged),
            max_model_len=self._max_length,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
        )
        try:
            self._llm = LLM(runner="pooling", pooler_config=pooler, **base)
        except TypeError:
            self._llm = LLM(task="classify", override_pooler_config=pooler, **base)

        self._tokenizer = AutoTokenizer.from_pretrained(merged)

    def _build_input_ids(self, query: str, document: str) -> list[int]:
        prefix_ids = self._tokenizer(
            f"query: Query: {query} document: ", add_special_tokens=False
        ).input_ids
        doc_ids = self._tokenizer(" " + document, add_special_tokens=False).input_ids
        tail_ids = self._tokenizer(" ", add_special_tokens=False).input_ids

        budget = self._max_length - len(prefix_ids) - len(tail_ids) - 1
        if budget > 0 and len(doc_ids) > budget:
            doc_ids = doc_ids[:budget]

        return prefix_ids + doc_ids + tail_ids + [self._tokenizer.eos_token_id]

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

        try:
            from vllm import TokensPrompt
        except ImportError:
            from vllm.inputs import TokensPrompt

        prompts = [
            TokensPrompt(prompt_token_ids=self._build_input_ids(query, doc))
            for doc in documents
        ]

        effective_batch = batch_size or self._batch_size or len(prompts)
        scores: list[float] = []
        for start in range(0, len(prompts), effective_batch):
            outputs = self._llm.classify(
                prompts[start : start + effective_batch],
                use_tqdm=show_progress_bar,
            )
            scores.extend(float(o.outputs.probs[0]) for o in outputs)
        return scores
