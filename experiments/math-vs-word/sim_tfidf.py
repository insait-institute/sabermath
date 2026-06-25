from datasets import Dataset
import json
import tqdm
from typing import List

from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sim_helpers import get_math_words_tokens


def tf_idf_scores(docs: List[List]):
    vectorizer = TfidfVectorizer(
        # Cast every token to a string so sklearn can sort the vocabulary
        tokenizer=lambda doc: [str(token) for token in doc],
        preprocessor=lambda x: x,
        token_pattern=None,
    )

    X = vectorizer.fit_transform(docs)

    target_vec = X[0]
    candidate_vecs = X[1:]

    scores = cosine_similarity(target_vec, candidate_vecs)[0]

    return scores


def calc_tfidf_sims(good_targets: Dataset, good_candidates: Dataset):

    output_path = "similarities/tf-idf.json"
    similarities_dict = {}

    for target in tqdm.tqdm(good_targets):

        target_id = target["id"]
        target_math_tokens, target_text_tokens = get_math_words_tokens(
            target["problem_math_expr"], target["problem_text_only"]
        )
        target_all_tokens = target_math_tokens + target_text_tokens

        candidates = []
        for i in target["candidates"]:
            candidate = good_candidates[i]
            cand_math_tokens, cand_words_tokens = get_math_words_tokens(
                candidate["problem_math_expr"] + candidate["solution_math_expr"],
                candidate["problem_text_only"] + candidate["solution_text_only"],
            )
            cand_all_tokens = cand_math_tokens + cand_words_tokens
            candidates.append(cand_all_tokens)

        math_docs = [target_math_tokens] + candidates
        text_docs = [target_text_tokens] + candidates
        all_docs = [target_all_tokens] + candidates

        math_scores = tf_idf_scores(math_docs)
        text_scores = tf_idf_scores(text_docs)
        all_scores = tf_idf_scores(all_docs)

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(mean(all_scores)),
            "pr_math_vs_candidates": float(mean(math_scores)),
            "pr_text_vs_candidates": float(mean(text_scores)),
        }

    with open(output_path, "w") as f:
        json.dump(similarities_dict, f)
