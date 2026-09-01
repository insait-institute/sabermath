from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
from datasets import load_dataset

from . import SIMILARITIES_DIR

DOMAIN_ORDER = [
    "Algebra",
    "Calculus and Analysis",
    "Combinatorics",
    "Geometry",
    "Number Theory",
]

GROUP_NAMES = DOMAIN_ORDER + ["All"]

FULL_KEY = "pr_full_vs_candidates"
MATH_KEY = "pr_math_vs_candidates"
TEXT_KEY = "pr_text_vs_candidates"

INSTRUCTIONS = ["p0", "p1", "p2", "p3"]
BASELINE_INSTRUCTION = "p0"

NON_EMBEDDING_METHODS = ["jaccard", "approach0", "tf-idf", "bm25"]

MODELS_WITH_EXPECTED_TIES = {
    "jaccard",
    "diver-grouprank-32b",
    "infly/inf-retriever-v1-pro",
    "inf-x-retriever",
    "retro-star-32b",
    "retro-star-32b-rewritten",
}


def normalize_instruction(value: str | None) -> str:
    """`None`/`0`/`p0` -> "p0"; `1`/`p1` -> "p1". Anything else raises."""
    if value is None:
        return BASELINE_INSTRUCTION
    key = str(value).strip().lower()
    if not key.startswith("p"):
        key = f"p{key}"
    if key not in INSTRUCTIONS:
        raise ValueError(
            f"instruction must be one of {INSTRUCTIONS} (or 0/1/2/3), got {value!r}"
        )
    return key


def file_stem(model_id: str) -> str:
    return model_id.replace("/", "_")


def similarity_path(
    model_id: str,
    instruction: str | None = None,
    similarities_dir: Path = SIMILARITIES_DIR,
) -> Path:
    """Where this (method, instruction) is stored. p0 has no suffix."""
    instruction = normalize_instruction(instruction)
    stem = file_stem(model_id)
    if instruction == BASELINE_INSTRUCTION:
        return Path(similarities_dir) / f"{stem}.json"
    return Path(similarities_dir) / f"{stem}__{instruction}.json"


def load_similarity_content(
    model_id: str,
    instruction: str | None = None,
    similarities_dir: Path = SIMILARITIES_DIR,
    *,
    baseline_fallback: bool = False,
) -> dict[str, dict[str, Any]]:
    path = similarity_path(model_id, instruction, similarities_dir)

    if not path.exists() and baseline_fallback:
        baseline = similarity_path(model_id, BASELINE_INSTRUCTION, similarities_dir)
        if baseline.exists():
            print(
                f"[~] No {path.name} for {model_id!r} - falling back to its "
                f"baseline file ({baseline.name})."
            )
            path = baseline

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find similarity file for model {model_id!r}: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def instructions_on_disk(
    model_ids: Iterable[str],
    similarities_dir: Path = SIMILARITIES_DIR,
) -> dict[str, list[str]]:
    found = {}
    for model_id in model_ids:
        instructions = [
            instruction
            for instruction in INSTRUCTIONS
            if similarity_path(model_id, instruction, similarities_dir).exists()
        ]
        if instructions:
            found[model_id] = instructions
    return found


def get_first_domain(domains) -> str:
    if isinstance(domains, (list, tuple)):
        if len(domains) == 0:
            raise ValueError("Encountered empty domains list.")
        return str(domains[0]).strip()
    if domains is None:
        raise ValueError("Encountered None domain.")
    return str(domains).strip()


def build_id_to_domain(
    *,
    all_content_ids: Iterable[str],
    targets_dataset_name: str,
) -> dict[str, str]:
    all_content_ids = {str(target_id) for target_id in all_content_ids}
    targets = load_dataset(targets_dataset_name, split="train").select_columns(
        ["id", "domains"]
    )

    id_to_domain = {}
    for target_id, domains in zip(targets["id"], targets["domains"]):
        target_id = str(target_id)
        if target_id in all_content_ids:
            id_to_domain[target_id] = get_first_domain(domains)

    missing_ids = [
        target_id for target_id in all_content_ids if target_id not in id_to_domain
    ]
    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} ids from the similarity files were not found "
            f"in the targets dataset. Examples: {missing_ids[:10]}"
        )
    return id_to_domain


@dataclass
class MathVsWordStats:
    percentages: dict[str, float]
    counts: dict[str, int]
    totals: dict[str, int]
    ties_skipped: int
    mean_pr: dict[str, float]

    @property
    def n_targets(self) -> int:
        return self.totals["All"] + self.ties_skipped


def aggregate_math_vs_words(
    content: Mapping[str, Mapping[str, Any]],
    *,
    model_id: str,
    id_to_domain: Mapping[str, str] | None = None,
) -> MathVsWordStats:
    counts = {group: 0 for group in GROUP_NAMES}
    totals = {group: 0 for group in GROUP_NAMES}
    pr_sums = {FULL_KEY: 0.0, MATH_KEY: 0.0, TEXT_KEY: 0.0}
    ties_skipped = 0
    n_seen = 0

    for target_id, row in content.items():
        target_id = str(target_id)

        missing_keys = [key for key in (TEXT_KEY, MATH_KEY) if key not in row]
        if missing_keys:
            raise KeyError(
                f"Model {model_id!r}, target {target_id!r} is missing keys: "
                f"{missing_keys}"
            )

        domain = None
        if id_to_domain is not None:
            if target_id not in id_to_domain:
                raise KeyError(
                    f"Model {model_id!r}, target {target_id!r} has no domain."
                )
            domain = str(id_to_domain[target_id]).strip()
            if domain not in DOMAIN_ORDER:
                raise ValueError(
                    f"Model {model_id!r}, target {target_id!r} has unknown domain "
                    f"{domain!r}. Expected one of: {DOMAIN_ORDER}"
                )

        text_score = float(row[TEXT_KEY])
        math_score = float(row[MATH_KEY])

        if not math.isfinite(text_score) or not math.isfinite(math_score):
            raise ValueError(
                f"Model {model_id!r}, target {target_id!r} has non-finite scores: "
                f"text={text_score}, math={math_score}"
            )

        n_seen += 1
        for key in pr_sums:
            value = row.get(key)
            if value is not None:
                pr_sums[key] += float(value)

        if text_score == math_score:
            if model_id in MODELS_WITH_EXPECTED_TIES:
                ties_skipped += 1
                continue
            raise ValueError(
                f"Model {model_id!r}, target {target_id!r} has tied text/math "
                f"scores: text={text_score}, math={math_score}"
            )

        totals["All"] += 1
        if domain is not None:
            totals[domain] += 1

        if math_score > text_score:
            counts["All"] += 1
            if domain is not None:
                counts[domain] += 1

    percentages = {
        group: (
            100.0 * counts[group] / totals[group]
            if totals[group] > 0
            else float("nan")
        )
        for group in GROUP_NAMES
    }

    mean_pr = {
        key: (total / n_seen if n_seen else float("nan"))
        for key, total in pr_sums.items()
    }

    return MathVsWordStats(
        percentages=percentages,
        counts=counts,
        totals=totals,
        ties_skipped=ties_skipped,
        mean_pr=mean_pr,
    )
