import pya0
import re
import tqdm
import json
from statistics import mean
from datasets import Dataset

from embed import get_top5_candidates
from sim_helpers import get_math_words_tokens


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1 & set2
    union = set1 | set2

    if not union:
        return 1.0  # both empty

    return len(intersection) / len(union)


def calc_jaccard_sims(good_targets: Dataset, good_candidates: Dataset):

    output_path = "similarities/jaccard.json"
    similarities_dict = {}

    for target in tqdm.tqdm(good_targets):

        target_id = target["id"]
        target_math_tokens, target_words_tokens = get_math_words_tokens(
            target["problem_math_expr"], target["problem_text_only"]
        )
        target_all_tokens = target_math_tokens + target_words_tokens

        top5_cand_idxs = get_top5_candidates(target)
        relevant_candidates = [good_candidates[i] for i in top5_cand_idxs]

        pr_full_sim = []
        pr_math_sim = []
        pr_cand_sim = []

        for candidate in relevant_candidates:
            cand_math_tokens, cand_words_tokens = get_math_words_tokens(
                candidate["problem_math_expr"] + candidate["solution_math_expr"],
                candidate["problem_text_only"] + candidate["solution_text_only"],
            )
            cand_all_tokens = cand_math_tokens + cand_words_tokens
            pr_math_sim.append(jaccard_similarity(target_math_tokens, cand_all_tokens))
            pr_cand_sim.append(jaccard_similarity(target_words_tokens, cand_all_tokens))
            pr_full_sim.append(jaccard_similarity(target_all_tokens, cand_all_tokens))

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(mean(pr_full_sim)),
            "pr_math_vs_candidates": float(mean(pr_math_sim)),
            "pr_text_vs_candidates": float(mean(pr_cand_sim)),
        }

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
