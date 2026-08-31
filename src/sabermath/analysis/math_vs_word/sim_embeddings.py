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
    # An instructed arm writes its OWN file, so an instructed run and a
    # prompt-free one can never overwrite each other.
    if instruction_key is not None:
        PATH_ID = f"{PATH_ID}__{instruction_key}"

    # STRIDED target split (i % shards == shard), matching
    # sabermath.runner.select_shard: every shard sees a mix of domains and
    # difficulties rather than one contiguous block, which matters here
    # because the targets dataset is domain-ordered and a contiguous block
    # would hand one shard all the geometry.
    #
    # Each shard writes its OWN file and never touches another's, so shards
    # are safe to run concurrently - which is the whole point, since
    # sim_embeddings rewrites its entire dict after every target and two
    # writers on one path would clobber each other. merge_sim_parts.py
    # stitches them back.
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

    # Both of these delegate to scripts/run_experiments.py, so the model is
    # built and scored exactly as the main experiment and run_dedup.py build
    # and score it - see load_models.py's header. instruction_key is passed
    # to get_model as well as to get_scores_kwargs because for three
    # families it changes the CONSTRUCTOR, not just the scoring kwargs.
    processor = get_model(model_id, instruction_key)

    # The per-model preprocessing protocol (chunk_to_context/context_length)
    # plus the vendor input envelope (query/document prompts and suffixes,
    # per-side API params). EmbeddingProcessor.get_scores applies the
    # envelope per side, before the vector cache.
    scores_kwargs = get_scores_kwargs(model_id, instruction_key)
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
    # and must NOT also get the generic wrap on top. The registry decides
    # this (its own wrap_instruction flag); we no longer guess.
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
    # Written via a temp file + os.replace rather than open(output_path, "w"),
    # which truncates. Two reasons, both hit in practice on 2026-08-26:
    #   * a reader (check_coverage.py) that lands
    #     inside the truncate/dump window sees an empty or half-written file
    #     and raises JSONDecodeError - observed against a running
    #     GTE-ModernColBERT job;
    #   * a job killed inside that window leaves a CORRUPT file, not a short
    #     one, so the resume path cannot recover it either.
    # os.replace is atomic within a filesystem. The name is dot-prefixed and
    # .tmp-suffixed so the region sync-back's --exclude="*.tmp" skips it,
    # matching run_experiments.py's own *.json.tmp convention.
    tmp_path = SIMILARITIES_DIR / f".{PATH_ID}.json.tmp"

    if not force_recalc:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                similarities_dict = json.load(f)

    print(f"===========Starting from idx {len(similarities_dict)}===========")

    # Batch any per-query preprocessing before scoring, mirroring
    # sabermath.runner's prefetch hook. Only the composed rewriter rows
    # implement prefetch_rewrites, and for them this is not an optimization
    # but a feasibility fix: get_scores() generates a MISSING rewrite one
    # query at a time, which leaves ~5 sequences in flight and runs ~10x
    # slower than one batched call (ReasonRewriterProcessor's PERFORMANCE
    # note). This sweep needs 3 fresh rewrites per target - the full, the
    # equation-only and the word-only variant, none of which the main
    # experiment has ever rewritten - so unbatched it is hours per model
    # rather than minutes.
    #
    # The texts handed over are EXACTLY what the loop below will score: the
    # same as_query() wrap and the same three variants, over the targets not
    # yet done, so an instructed arm prefetches its own wrapped strings and a
    # resumed run does not regenerate what it already has.
    prefetch = getattr(processor, "prefetch_rewrites", None)
    if prefetch is not None:
        pending, seen = [], set()
        # Index one row at a time, exactly as the scoring loop below does.
        # NOT good_targets[start:] - a datasets.Dataset slices COLUMNAR and
        # returns dict[str, list], so iterating the slice yields column NAMES
        # and t[field] raises "string indices must be integers".
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
