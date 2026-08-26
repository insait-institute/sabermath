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
    instruction_key: str | None = None,
):

    PATH_ID = model_id.replace("/", "_")
    # An instructed arm writes its OWN file. The unsuffixed name stays the
    # prompt-free run every published similarities/ file already holds, so
    # adding an ablation can never overwrite it.
    if instruction_key is not None:
        PATH_ID = f"{PATH_ID}__{instruction_key}"

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
    scores_kwargs = get_scores_kwargs(model_id, instruction_key)
    if scores_kwargs:
        print(f"[~] Extra get_scores() kwargs for {model_id}: {scores_kwargs}")

    # The instruction is applied to the QUERY side only, through the same
    # helper sabermath.benchmark.evaluate() uses, so an instructed math-vs-word
    # query is byte-identical to an instructed main-benchmark query. All three
    # target representations are wrapped: the comparison is between full,
    # equation-only and word-only content of the SAME instructed query.
    instruction_text = None
    if instruction_key is not None:
        from sabermath.instructions import INSTRUCTIONS, format_instructed_query

        instruction_text = INSTRUCTIONS[instruction_key]
        print(f"[~] Instruction {instruction_key}: {instruction_text!r}")

    def as_query(text: str) -> str:
        if instruction_text is None:
            return text
        return format_instructed_query(instruction_text, text)

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

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
