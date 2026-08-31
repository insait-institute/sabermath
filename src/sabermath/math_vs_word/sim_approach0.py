import json
import os
import shutil
import tempfile
from statistics import mean

import tqdm
from datasets import Dataset

from ..processors.approach0_processor import build_pya0_index, search_pya0_index
from . import SIMILARITIES_DIR
from .embed import get_top5_candidates

# This module used to carry its own copy of the pya0 tokenizer, and the copies
# had drifted: this one wraps each LaTeX pattern in a capturing group, so the
# $...$ / \[...\] markers stay INSIDE the tex chunk and pya0 receives
# "[imath]$x$[/imath]" instead of "[imath]x[/imath]". The processor's variant -
# the one the benchmark's approach0 rows use - strips them.
#
# Set False to adopt the processor's tokenization. That CHANGES the committed
# results/math_vs_word/similarities/approach0.json numbers, so it is a
# deliberate decision, not a cleanup.
KEEP_DELIMITERS = True


def approach0_scores(
    query: str,
    documents: list[str],
    *,
    keep_index_dir: str | None = None,
    show_progress_bar: bool = True,
) -> list[float] | None:
    if len(documents) == 0:
        return []

    if keep_index_dir is None:
        index_dir = tempfile.mkdtemp(prefix="pya0_problem_index_")
        cleanup = True
    else:
        index_dir = keep_index_dir
        os.makedirs(index_dir, exist_ok=True)
        cleanup = False

    try:
        build_pya0_index(documents, index_dir, keep_delimiters=KEEP_DELIMITERS)

        hits = search_pya0_index(
            problems=documents,
            target=query,
            index_dir=index_dir,
            topk=len(documents),
            keep_delimiters=KEEP_DELIMITERS,
        )

        if hits is None:
            return None

        by_index = {h["index"]: h for h in hits if 0 <= h["index"] < len(documents)}

        ranked = []
        for i, document in enumerate(documents):
            h = by_index.get(i)
            if h is None:
                ranked.append(0.0)
            else:
                ranked.append(h["score"])

        return ranked

    finally:
        if cleanup:
            shutil.rmtree(index_dir, ignore_errors=True)


def calc_approach0_sims(good_targets: Dataset, good_candidates: Dataset):

    SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SIMILARITIES_DIR / "approach0.json"

    similarities_dict = {}

    for target in tqdm.tqdm(good_targets):

        target_id = target["id"]
        cands_idxs = get_top5_candidates(target)

        query_math = target["problem_math_expr"]
        query_text = target["problem_text_only"]
        query_full = target["problem_fixed"]

        candidates = [
            good_candidates[i]["problem_fixed"] + good_candidates[i]["solution_fixed"]
            for i in cands_idxs
        ]

        math_scores = approach0_scores(query_math, candidates)
        text_scores = approach0_scores(query_text, candidates)
        full_scores = approach0_scores(query_full, candidates)

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(mean(full_scores)),
            "pr_math_vs_candidates": float(mean(math_scores)),
            "pr_text_vs_candidates": float(mean(text_scores)),
        }

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
