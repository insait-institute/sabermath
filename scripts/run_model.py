import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import multiprocessing as mp

import sabermath
from sabermath.processors import (
    Approach0Processor,
    BM25Processor,
    GoogleProcessor,
    JaccardProcessor,
    OpenAIProcessor,
    TfidfProcessor,
)


OPENAI_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]


GOOGLE_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]


LEGACY_MODELS = [
    "bm25",
    "tf-idf",
    "approach0",
    "jaccard",
]


ModelKind = Literal["hf", "google", "openai", "legacy"]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: ModelKind
    name: str


def ensure_dir(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def make_output_path(model_name: str, dir: str | Path) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    return Path(dir) / f"{safe}.json"


def normalize_legacy_name(name: str) -> str | None:
    normalized = name.strip().lower().replace("_", "-")

    aliases = {
        "bm25": "bm25",
        "tf-idf": "tf-idf",
        "tfidf": "tf-idf",
        # Common typo. Keep it supported because older notes/scripts used it.
        "td-idf": "tf-idf",
        "tdidf": "tf-idf",
        "approach0": "approach0",
        "approach-0": "approach0",
        "jaccard": "jaccard",
    }

    return aliases.get(normalized)


def make_legacy_processor(name: str, init_kwargs: dict):
    canonical = normalize_legacy_name(name)

    if canonical == "bm25":
        return BM25Processor(**init_kwargs)
    if canonical == "tf-idf":
        return TfidfProcessor(**init_kwargs)
    if canonical == "approach0":
        return Approach0Processor(**init_kwargs)
    if canonical == "jaccard":
        return JaccardProcessor(**init_kwargs)

    raise ValueError(f"Unknown legacy model: {name}")


def make_api_or_legacy_processor(kind: ModelKind, name: str, init_kwargs: dict):
    if kind == "google":
        return GoogleProcessor(name, **init_kwargs)
    if kind == "openai":
        return OpenAIProcessor(name, **init_kwargs)
    if kind == "legacy":
        return make_legacy_processor(name, init_kwargs)

    raise ValueError(f"Cannot build processor for kind: {kind}")


def expand_model_specs(name: str) -> list[ModelSpec]:
    raw = name.strip()
    key = raw.lower().replace("_", "-")

    google_specs = [
        ModelSpec(f"google/{model}", "google", model) for model in GOOGLE_MODELS
    ]
    openai_specs = [
        ModelSpec(f"openai/{model}", "openai", model) for model in OPENAI_MODELS
    ]
    legacy_specs = [ModelSpec(model, "legacy", model) for model in LEGACY_MODELS]

    groups: dict[str, list[ModelSpec]] = {
        "google": google_specs,
        "gemini": google_specs,
        "openai": openai_specs,
        "api": google_specs + openai_specs,
        "apis": google_specs + openai_specs,
        "legacy": legacy_specs,
        "old": legacy_specs,
        "baseline": legacy_specs,
        "baselines": legacy_specs,
        "research": legacy_specs,
        "non-neural": legacy_specs,
        "special": google_specs + openai_specs + legacy_specs,
    }

    if key in groups:
        return groups[key]

    if raw.startswith("google/"):
        return [ModelSpec(raw, "google", raw.split("/", 1)[1])]
    if raw.startswith("openai/"):
        return [ModelSpec(raw, "openai", raw.split("/", 1)[1])]

    if raw in GOOGLE_MODELS:
        return [ModelSpec(f"google/{raw}", "google", raw)]
    if raw in OPENAI_MODELS:
        return [ModelSpec(f"openai/{raw}", "openai", raw)]

    legacy_name = normalize_legacy_name(raw)
    if legacy_name is not None:
        return [ModelSpec(legacy_name, "legacy", legacy_name)]

    return [ModelSpec(raw, "hf", raw)]


def cache_path_for(model_label: str, cache_directory: str | Path) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._*-]", "_", model_label)
    return str(Path(cache_directory) / f"{safe_name}.pk")


def _run_isolated(
    name: str,
    use_vllm: bool,
    save_dir: str | Path,
    init_kwargs: dict,
    encode_kwargs: dict,
    dcg_variant: str,
    cache: str | None,
):
    try:
        print(f'\n\n[!!]\n\n[~] Running model "{name}"...\n\n[!!]\n\n')
        report = sabermath.evaluate(
            name,
            use_vllm=use_vllm,
            cache_path=cache,
            dcg_variant=dcg_variant,
            scores_kwargs=encode_kwargs,
            init_kwargs=init_kwargs,
        )
        obj = report.to_dict()
    except Exception as e:
        print(f"\n\n[!!] Error: {e} [!!]\n\n")
        obj = {
            "model": name,
            "error": str(e),
        }

    filepath = make_output_path(name, save_dir)
    ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_processor_isolated(
    label: str,
    kind: ModelKind,
    name: str,
    save_dir: str | Path,
    init_kwargs: dict,
    encode_kwargs: dict,
    dcg_variant: str,
    cache: str | None,
):
    try:
        print(f'\n\n[!!]\n\n[~] Running model "{label}"...\n\n[!!]\n\n')
        processor = make_api_or_legacy_processor(kind, name, init_kwargs)
        report = sabermath.evaluate(
            processor,
            cache_path=cache,
            dcg_variant=dcg_variant,
            scores_kwargs=encode_kwargs,
            init_kwargs={},
        )
        obj = report.to_dict()
        obj["run_label"] = label
    except Exception as e:
        print(f"\n\n[!!] Error: {e} [!!]\n\n")
        obj = {
            "model": name,
            "run_label": label,
            "processor": kind,
            "error": str(e),
        }

    filepath = make_output_path(label, save_dir)
    ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_isolated_preloaded(
    model,
    model_name: str,
    save_dir: str | Path,
    init_kwargs: dict,
    encode_kwargs: dict,
    dcg_variant: str,
    cache: str | None,
):
    try:
        print(f'\n\n[!!]\n\n[~] Running model "{model_name}"...\n\n[!!]\n\n')
        report = sabermath.evaluate(
            model,
            cache_path=cache,
            dcg_variant=dcg_variant,
            scores_kwargs=encode_kwargs,
            init_kwargs={},
        )
        obj = report.to_dict()
    except Exception as e:
        print(f"\n\n[!!] Error: {e} [!!]\n\n")
        obj = {
            "model": model_name,
            "error": str(e),
        }

    filepath = make_output_path(model_name, save_dir)
    ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    # General Config
    parser.add_argument(
        "name",
        type=str,
        help=(
            "Model to evaluate. In addition to HuggingFace names, this accepts "
            "google/<model>, openai/<model>, bm25, tf-idf/td-idf, approach0, "
            "jaccard, or groups: google, openai, api/apis, legacy/old/baselines, "
            "special."
        ),
    )

    parser.add_argument(
        "--save-to", type=str, default="results", help="Where to save results"
    )

    # Benchmark & Computation Config

    parser.add_argument(
        "--driver",
        type=str,
        choices=["vllm", "st"],
        default="st",
        help="Which driver to use for running HuggingFace models (vLLM or SentenceTransformers)",
    )

    parser.add_argument(
        "--dcg-variant",
        type=str,
        choices=["exponent", "linear"],
        default="exponent",
        help="Which DCG variant to use when evaluating (linear or exponent)",
    )

    parser.add_argument(
        "--encode-kwargs",
        type=json.loads,
        default=None,
        help="Additional arguments to pass to the score/encoding processor",
    )

    parser.add_argument(
        "--init-kwargs",
        type=json.loads,
        default=None,
        help="Additional arguments to pass to the model/processor initializer",
    )

    # Cache Parameters
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Activate cache",
    )

    parser.add_argument(
        "--cache-directory",
        type=str,
        default=".cache",
        help="Directory of cached vectors",
    )

    args = parser.parse_args()

    encode_kwargs = args.encode_kwargs or {}
    init_kwargs = args.init_kwargs or {}
    use_vllm = args.driver == "vllm"

    ctx = mp.get_context("spawn")

    chunk_to_context = False
    if args.name.endswith("*"):
        chunk_to_context = True
        args.name = args.name[:-1]

    specs = expand_model_specs(args.name)

    if chunk_to_context:
        non_hf = [spec.label for spec in specs if spec.kind != "hf"]
        if non_hf:
            print(
                "[WARNING] '*' chunk-to-context mode only applies to "
                f"HuggingFace models. Ignoring it for: {', '.join(non_hf)}"
            )
        else:
            if use_vllm:
                print(
                    "[WARNING] Models requiring vector accumulation "
                    "(marked with *) are currently not supported by vLLM. "
                    "Switching to SentenceTransformers."
                )
            use_vllm = False
            encode_kwargs["chunk_to_context"] = True

    failures: list[tuple[str, int | None]] = []

    for spec in specs:
        cache_file = (
            cache_path_for(spec.label, args.cache_directory) if args.cache else None
        )

        if spec.kind == "hf":
            run_params = {
                "target": _run_isolated,
                "args": (
                    spec.name,
                    use_vllm,
                    args.save_to,
                    init_kwargs,
                    encode_kwargs,
                    args.dcg_variant,
                    cache_file,
                ),
            }
        else:
            run_params = {
                "target": _run_processor_isolated,
                "args": (
                    spec.label,
                    spec.kind,
                    spec.name,
                    args.save_to,
                    init_kwargs,
                    encode_kwargs,
                    args.dcg_variant,
                    cache_file,
                ),
            }

        p = ctx.Process(**run_params)
        p.start()
        p.join()

        if p.exitcode != 0:
            filepath = make_output_path(spec.label, args.save_to)

            ensure_dir(filepath)

            obj = {
                "model": spec.name,
                "run_label": spec.label,
                "ok": False,
                "error": f"Process exited with code {p.exitcode}",
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

            print(f"[!!] Process failed for {spec.label}; saved error to {filepath}")
            failures.append((spec.label, p.exitcode))

    if failures:
        labels = ", ".join(label for label, _ in failures)
        raise SystemExit(f"Failed runs: {labels}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
