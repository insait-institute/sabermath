import os
import json
import tqdm
from datasets import Dataset
from statistics import mean

from embed import get_top5_candidates
from load_models import (
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
    legacy: bool = False,
):

    PATH_ID = model_id.replace("/", "_")
    # An instructed arm writes its OWN file, and so does a legacy-protocol
    # run - same reason run_rerankers.py tags its legacy results (see its
    # --legacy help): the two protocols must never overwrite each other.
    if instruction_key is not None:
        PATH_ID = f"{PATH_ID}__{instruction_key}"
    if legacy:
        PATH_ID = f"{PATH_ID}__legacy"

    print(f"============== {model_id} ==============")

    # Both of these delegate to scripts/run_rerankers.py, so the model is
    # built and scored exactly as the main experiment and run_dedup.py build
    # and score it - see load_models.py's header. instruction_key is passed
    # to get_model as well as to get_scores_kwargs because for three
    # families it changes the CONSTRUCTOR, not just the scoring kwargs.
    processor = get_model(model_id, instruction_key, legacy=legacy)

    # The per-model preprocessing protocol (chunk_to_context/context_length)
    # plus the vendor input envelope (query/document prompts and suffixes,
    # per-side API params). EmbeddingProcessor.get_scores applies the
    # envelope per side, before the vector cache.
    scores_kwargs = get_scores_kwargs(model_id, instruction_key, legacy=legacy)
    if scores_kwargs:
        print(f"[~] get_scores() kwargs for {model_id}: {scores_kwargs}")
    assert_envelope_supported(model_id, processor, scores_kwargs)

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

    # Some models carry the instruction through their own mechanism - the
    # qwen3-reranker <Instruct> slot, ReasonIR's masked-prefix encode kwarg -
    # and must NOT also get the generic wrap on top. run_rerankers decides
    # this (its own wrap_instruction flag); we no longer guess.
    wrap = instruction_text is not None and wraps_instruction(
        model_id, instruction_key, legacy=legacy
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

    output_path = f"similarities/{PATH_ID}.json"
    # Written via a temp file + os.replace rather than open(output_path, "w"),
    # which truncates. Two reasons, both hit in practice on 2026-08-26:
    #   * a reader (scripts/compare_recalc.py, check_coverage.py) that lands
    #     inside the truncate/dump window sees an empty or half-written file
    #     and raises JSONDecodeError - observed against a running
    #     GTE-ModernColBERT job;
    #   * a job killed inside that window leaves a CORRUPT file, not a short
    #     one, so the resume path cannot recover it either.
    # os.replace is atomic within a filesystem. The name is dot-prefixed and
    # .tmp-suffixed so the region sync-back's --exclude="*.tmp" skips it,
    # matching run_rerankers.py's own *.json.tmp convention.
    tmp_path = f"similarities/.{PATH_ID}.json.tmp"

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

        with open(tmp_path, "w") as f:
            json.dump(similarities_dict, f)
        os.replace(tmp_path, output_path)
