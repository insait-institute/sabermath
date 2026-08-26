import os
import json
import tqdm
from datasets import Dataset
from statistics import mean

from embed import get_top5_candidates
from load_models import get_model, get_scores_kwargs


def calc_embedding_sims(
    model_id: str,
    good_targets: Dataset,
    good_candidates: Dataset,
    force_recalc: bool = False,
):

    PATH_ID = model_id.replace("/", "_")

    print(f"============== {model_id} ==============")

    # get_model() returns an actual sabermath.processors instance
    # (SentenceTransformersProcessor / VLLMProcessor / GoogleProcessor) -
    # scored below via its own .get_scores(), the exact same call
    # sabermath.benchmark.evaluate_task() makes in production.
    processor = get_model(model_id)

    # Empty for every model except the RaDeR bi-encoder family, which needs
    # chunk_to_context/context_length forwarded to get_scores() to match
    # scripts/run_rerankers.py's own protocol (see load_models.py's
    # get_scores_kwargs docstring).
    scores_kwargs = get_scores_kwargs(model_id)
    if scores_kwargs:
        print(f"[~] Extra get_scores() kwargs for {model_id}: {scores_kwargs}")

    similarities_dict = {}

    output_path = f"similarities/{PATH_ID}.json"

    if not force_recalc:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                similarities_dict = json.load(f)

    print(f"===========Starting from idx {len(similarities_dict)}===========")

    for _ in tqdm.tqdm(range(len(similarities_dict), len(good_targets))):

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

        # Same call shape as sabermath.benchmark.evaluate_task():
        # processor.get_scores(query, documents, show_progress_bar=False) -
        # one call per query variant, all three against the same 5
        # candidates. get_scores()'s own default caching (check_cache=
        # update_cache=True) means the 5 candidate embeddings are computed
        # once (on the first of the three calls below) and reused for the
        # other two, same as it would be reused across repeated targets that
        # happen to share a candidate.
        pr_full_vs_candidates = mean(
            processor.get_scores(
                target_problem_full,
                candidates_compare,
                show_progress_bar=False,
                **scores_kwargs,
            )
        )
        pr_math_vs_candidates = mean(
            processor.get_scores(
                target_problem_math,
                candidates_compare,
                show_progress_bar=False,
                **scores_kwargs,
            )
        )
        pr_text_vs_candidates = mean(
            processor.get_scores(
                target_problem_text,
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

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
