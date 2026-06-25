import os
import json
import numpy as np
import tqdm
from datasets import Dataset
from statistics import mean

from embed import get_top5_candidates, get_embeddings
from load_models import get_model


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calc_embedding_sims(
    model_id: str,
    good_targets: Dataset,
    good_candidates: Dataset,
    force_recalc: bool = False,
):

    PATH_ID = model_id.replace("/", "_")

    print(f"============== {model_id} ==============")

    model_dict = get_model(model_id)

    model = model_dict["model"]
    model_type = model_dict["type"]

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

        to_embed = [
            target_problem_full,
            target_problem_math,
            target_problem_text,
        ] + candidates_compare

        problem_full, problem_math, problem_text, c1, c2, c3, c4, c5 = get_embeddings(
            model_name=model_id,
            model=model,
            type=model_type,
            texts=to_embed,
            batch_size=10,
        )

        pr_full_vs_candidates = mean(
            [cosine_similarity(problem_full, c) for c in [c1, c2, c3, c4, c5]]
        )
        pr_math_vs_candidates = mean(
            [cosine_similarity(problem_math, c) for c in [c1, c2, c3, c4, c5]]
        )
        pr_text_vs_candidates = mean(
            [cosine_similarity(problem_text, c) for c in [c1, c2, c3, c4, c5]]
        )

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(pr_full_vs_candidates),
            "pr_math_vs_candidates": float(pr_math_vs_candidates),
            "pr_text_vs_candidates": float(pr_text_vs_candidates),
        }

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
