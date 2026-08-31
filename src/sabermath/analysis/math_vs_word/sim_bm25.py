import json
import tqdm
from statistics import mean
from datasets import Dataset
from rank_bm25 import BM25Okapi

from embed import get_top5_candidates
from sim_helpers import get_math_words_tokens


def calc_bm25_sims(good_targets: Dataset, good_candidates: Dataset):

    output_path = "similarities/bm25.json"
    similarities_dict = {}

    for target in tqdm.tqdm(good_targets):

        target_id = target["id"]
        target_math_tokens, target_words_tokens = get_math_words_tokens(
            target["problem_math_expr"], target["problem_text_only"]
        )
        target_all_tokens = target_math_tokens + target_words_tokens

        top5_cand_idxs = get_top5_candidates(target)
        relevant_candidates = [good_candidates[i] for i in top5_cand_idxs]

        cand_all_tokens_list = []
        for candidate in relevant_candidates:
            cand_math_tokens, cand_words_tokens = get_math_words_tokens(
                candidate["problem_math_expr"] + candidate["solution_math_expr"],
                candidate["problem_text_only"] + candidate["solution_text_only"],
            )
            cand_all_tokens_list.append(cand_math_tokens + cand_words_tokens)

        # Same corpus (the 5 candidates, each represented by their full
        # problem+solution tokens) for all 3 query variants below - mirrors
        # sim_jaccard.py/sim_tfidf.py, where only the query side changes
        # between full/math/text and the candidate side is always "full".
        bm25 = BM25Okapi(cand_all_tokens_list)

        full_scores = bm25.get_scores(target_all_tokens)
        math_scores = bm25.get_scores(target_math_tokens)
        text_scores = bm25.get_scores(target_words_tokens)

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(mean(full_scores)),
            "pr_math_vs_candidates": float(mean(math_scores)),
            "pr_text_vs_candidates": float(mean(text_scores)),
        }

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
