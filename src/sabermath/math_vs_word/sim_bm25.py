import json
import tqdm
from statistics import mean
from datasets import Dataset
from rank_bm25 import BM25Okapi

from .embed import get_top5_candidates
from .sim_helpers import get_math_words_tokens
from . import SIMILARITIES_DIR


def calc_bm25_sims(good_targets: Dataset, good_candidates: Dataset):

    SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SIMILARITIES_DIR / "bm25.json"
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
