from sabermath.registry import RADER_BIENCODER_MODELS
from sabermath.processors import SentenceTransformersProcessor
from . import SIMILARITIES_DIR
from .models import ALLOWED_MODELS

_NO_EXPERIMENT_KEY = {"microsoft/codebert-base"}
_TENSOR_PARALLEL_SIZE = 1
_SAVE_DIR = str(SIMILARITIES_DIR)


def _model_key_by_id() -> dict:
    from sabermath import registry as rr

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
    from sabermath import registry as rr

    if model_id in rr.ALL_MODEL_KEYS:
        return model_id
    key = _model_key_by_id().get(model_id)
    if key is None:
        candidate = model_id.rsplit("/", 1)[-1].lower()
        if candidate in rr.ALL_MODEL_KEYS:
            return candidate
    if key is None:
        raise KeyError(
            f"{model_id} has no scripts/run_experiments.py model key, so its "
            "input envelope and instruction handling are unknown. Add it to "
            "that file's spec dicts rather than special-casing it here."
        )
    return key


def _instruction_args(instruction_key: str | None) -> tuple[str, str | None]:
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
) -> dict:
    if model_id in _NO_EXPERIMENT_KEY:
        return {}

    from sabermath import registry as rr

    key, instruction_text = _instruction_args(instruction_key)
    kwargs, _ = rr.prompt_scores_kwargs(
        model_key_for(model_id), instruction_text
    )
    return kwargs


def wraps_instruction(
    model_id: str,
    instruction_key: str | None = None,
) -> bool:
    if model_id in _NO_EXPERIMENT_KEY:
        return True

    from sabermath import registry as rr

    key, instruction_text = _instruction_args(instruction_key)
    model_key = model_key_for(model_id)
    _, wrap = rr.prompt_scores_kwargs(model_key, instruction_text)
    if model_key in rr.QWEN3_RERANKER_REPOS:
        return False
    return wrap


def assert_envelope_supported(model_id: str, processor, scores_kwargs: dict) -> None:
    if model_id in _NO_EXPERIMENT_KEY:
        return

    from sabermath import registry as rr

    rr._assert_envelope_supported(model_key_for(model_id), processor, scores_kwargs)


def get_model(
    MODEL_ID: str,
    instruction_key: str | None = None,
):
    if MODEL_ID in _NO_EXPERIMENT_KEY:
        print(
            f"Loading {MODEL_ID} via SentenceTransformersProcessor."
            f"from_huggingface (no the registry key - see _NO_EXPERIMENT_KEY)..."
        )
        return SentenceTransformersProcessor.from_huggingface(
            MODEL_ID, trust_remote_code=True
        )

    from sabermath import registry as rr

    key, _ = _instruction_args(instruction_key)
    model_key = model_key_for(MODEL_ID)
    protocol = "canonical"
    print(
        f"Loading {MODEL_ID} as the registry model key {model_key!r} "
        f"(instruction {key}, {protocol} protocol) via "
        f"the registry's processor builder..."
    )
    return rr.build_processor(
        model_key, key, _TENSOR_PARALLEL_SIZE, _SAVE_DIR
    )


assert set(RADER_BIENCODER_MODELS.values()) <= set(ALLOWED_MODELS), (
    "RADER_BIENCODER_MODELS and math_vs_word.models.ALLOWED_MODELS disagree"
)
