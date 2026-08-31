import os
import json
import tqdm
from datasets import Dataset
from statistics import mean

from . import SIMILARITIES_DIR
from .embed import get_top5_candidates
from .load_models import (
    assert_envelope_supported,
    get_model,
    get_scores_kwargs,
    wraps_instruction,
)


def calc_embedding_sims(
    model_id: str,
    good_targets: Dataset,
    good_candidates: Dataset,
    force_recalc: bool = False,
    instruction_key: str | None = None,
    query_shard: int | None = None,
    query_shards: int | None = None,
):

    PATH_ID = model_id.replace("/", "_")
    if instruction_key is not None:
        PATH_ID = f"{PATH_ID}__{instruction_key}"

    if (query_shard is None) != (query_shards is None):
        raise ValueError("query_shard and query_shards must be given together")
    if query_shards is not None:
        if not (0 <= query_shard < query_shards):
            raise ValueError(
                f"query_shard must be in [0, {query_shards}), got {query_shard}"
            )
        target_idxs = [
            i for i in range(len(good_targets)) if i % query_shards == query_shard
        ]
        if not target_idxs:
            raise ValueError(
                f"Shard {query_shard}/{query_shards} selected no targets."
            )
        PATH_ID = f"{PATH_ID}__shard{query_shard}of{query_shards}"
        print(
            f"[~] Target shard {query_shard}/{query_shards}: "
            f"{len(target_idxs)} of {len(good_targets)} targets."
        )
    else:
        target_idxs = list(range(len(good_targets)))

    print(f"============== {model_id} ==============")

    processor = get_model(model_id, instruction_key)

    scores_kwargs = get_scores_kwargs(model_id, instruction_key)
    if scores_kwargs:
        print(f"[~] get_scores() kwargs for {model_id}: {scores_kwargs}")
    assert_envelope_supported(model_id, processor, scores_kwargs)

    instruction_text = None
    if instruction_key is not None:
        from sabermath.instructions import INSTRUCTIONS, format_instructed_query

        instruction_text = INSTRUCTIONS[instruction_key]
        print(f"[~] Instruction {instruction_key}: {instruction_text!r}")

    wrap = instruction_text is not None and wraps_instruction(
        model_id, instruction_key
    )
    if instruction_text is not None and not wrap:
        print(
            f"[~] {model_id} carries the instruction through its own "
            f"mechanism - not applying the generic Instruct:/Query: wrap."
        )

    def as_query(text: str) -> str:
        if not wrap:
            return text
        return format_instructed_query(instruction_text, text)

    similarities_dict = {}

    SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SIMILARITIES_DIR / f"{PATH_ID}.json"
    tmp_path = SIMILARITIES_DIR / f".{PATH_ID}.json.tmp"

    if not force_recalc:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                similarities_dict = json.load(f)

    print(f"===========Starting from idx {len(similarities_dict)}===========")

    prefetch = getattr(processor, "prefetch_rewrites", None)
    if prefetch is not None:
        pending, seen = [], set()
        for i in target_idxs[len(similarities_dict):]:
            t = good_targets[i]
            for field in ("problem_fixed", "problem_math_expr", "problem_text_only"):
                text = as_query(t[field])
                if text not in seen:
                    seen.add(text)
                    pending.append(text)
        if pending:
            print(f"[~] Prefetching rewrites for {len(pending)} query texts "
                  f"in one batched call...")
            prefetch(pending)

    todo = target_idxs[len(similarities_dict):]
    for _ in tqdm.tqdm(todo, initial=len(similarities_dict), total=len(target_idxs)):

        target = good_targets[_]
        target_id = target["id"]

        target_problem_full = target["problem_fixed"]
        target_problem_math = target["problem_math_expr"]
        target_problem_text = target["problem_text_only"]

        top5_cand_idxs = get_top5_candidates(target)
        candidates_compare = [
            good_candidates[s]["problem_fixed"] + good_candidates[s]["solution_fixed"]
            for s in top5_cand_idxs
        ]

        pr_full_vs_candidates = mean(
            processor.get_scores(
                as_query(target_problem_full),
                candidates_compare,
                show_progress_bar=False,
                **scores_kwargs,
            )
        )
        pr_math_vs_candidates = mean(
            processor.get_scores(
                as_query(target_problem_math),
                candidates_compare,
                show_progress_bar=False,
                **scores_kwargs,
            )
        )
        pr_text_vs_candidates = mean(
            processor.get_scores(
                as_query(target_problem_text),
                candidates_compare,
                show_progress_bar=False,
                **scores_kwargs,
            )
        )

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(pr_full_vs_candidates),
            "pr_math_vs_candidates": float(pr_math_vs_candidates),
            "pr_text_vs_candidates": float(pr_text_vs_candidates),
        }

        with open(tmp_path, "w") as f:
            json.dump(similarities_dict, f)
        os.replace(tmp_path, output_path)
