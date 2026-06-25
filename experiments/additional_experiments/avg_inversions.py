import argparse
import csv
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from loguru import logger
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from llmteach.api import APIQuery
from llmteach.executor import QueryExecutor
from llmteach.postprocess import fix_thinking


NUM_PLAYERS_DEFAULT = 150
NUM_ROUNDS_DEFAULT = 50
CHECKPOINT_EVERY_DEFAULT = 5
BATCH_SIZE_DEFAULT = 200


def extract_boxed_number(s: str) -> int:
    match = re.search(r"\\boxed\{\s*([12])\s*\}", s)
    if not match:
        raise ValueError("No valid \\boxed{1} or \\boxed{2} found")
    return int(match.group(1))


def has_valid_relevance_scores_full(
    relevance_scores_full: Optional[List[float]],
    num_players: int,
) -> bool:
    """
    Returns True iff relevance_scores_full exists and has at least num_players valid scores.
    """

    if relevance_scores_full is None:
        return False

    if not isinstance(relevance_scores_full, list):
        return False

    if len(relevance_scores_full) < num_players:
        return False

    for x in relevance_scores_full[:num_players]:
        if x is None:
            return False

        try:
            x_float = float(x)
        except (TypeError, ValueError):
            return False

        if math.isnan(x_float):
            return False

    return True


def next_pairs(
    scores: Dict[str, int],
    pairs_played: List[Tuple[str, str]],
    num_players: int,
) -> List[Tuple[str, str]]:
    """
    Generates the next num_players / 2 Swiss-style pairs.

    Pairing logic follows your original code:
    - random shuffle first,
    - sort by current Swiss score,
    - greedily pair nearby players,
    - avoid repeated matches.
    """

    items = list(scores.keys())
    random.shuffle(items)

    ids_sorted = sorted(items, key=lambda x: scores[x])

    begin, end = 0, num_players

    used = set()
    pairings = []
    idxs_not_paired = []

    played_set = set(tuple(sorted(pair)) for pair in pairs_played)

    for i, player in enumerate(ids_sorted):
        if i < begin or i >= end:
            continue

        if player in used:
            continue

        for opponent in ids_sorted[i + 1 : end]:
            if opponent in used:
                continue

            new_pair = tuple(sorted((player, opponent)))

            if new_pair in played_set:
                continue

            pairings.append(new_pair)
            used.add(player)
            used.add(opponent)
            break

        if player not in used:
            idxs_not_paired.append(i)

    not_used = set(ids_sorted[begin:end]).difference(used)

    if len(not_used) != 0:
        print(
            f"WARNING: {len(not_used)} people not playing: "
            f"{[x for x in idxs_not_paired]}"
        )

    return pairings


def compute_bt_ratings(df: pd.DataFrame, C: float = 1.0, **kwargs) -> pd.Series:
    """
    Fits a Bradley-Terry model using logistic regression.

    Required columns:
    - model_a
    - model_b
    - winner

    The winner column must contain the actual winning model id, not 1 or 2.
    """

    required_cols = ["model_a", "model_b", "winner"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    if len(df) == 0:
        raise ValueError("Cannot fit Bradley-Terry model on empty dataframe.")

    players = pd.unique(df[["model_a", "model_b"]].values.ravel("K"))
    player_map = {name: i for i, name in enumerate(players)}
    num_players = len(players)

    num_matches = len(df)

    X = np.zeros((num_matches, num_players))
    y = np.zeros(num_matches)

    for i, row in enumerate(df.itertuples(index=False)):
        idx_a = player_map[row.model_a]
        idx_b = player_map[row.model_b]

        X[i, idx_a] = 1
        X[i, idx_b] = -1

        if row.winner == row.model_a:
            y[i] = 1
        elif row.winner == row.model_b:
            y[i] = 0
        else:
            raise ValueError(
                f"Winner '{row.winner}' does not match "
                f"model_a '{row.model_a}' or model_b '{row.model_b}'."
            )

    sample_weights = np.ones(len(y))

    # Prevent LogisticRegression from crashing if all labels are the same.
    if np.all(y == 0):
        X = np.vstack([X, np.zeros(X.shape[1])])
        y = np.append(y, 1)
        sample_weights = np.append(sample_weights, 0.0)
    elif np.all(y == 1):
        X = np.vstack([X, np.zeros(X.shape[1])])
        y = np.append(y, 0)
        sample_weights = np.append(sample_weights, 0.0)

    lr = LogisticRegression(fit_intercept=False, C=C, **kwargs)
    lr.fit(X, y, sample_weight=sample_weights)

    coefficients = lr.coef_[0]
    ratings = pd.Series(coefficients[:num_players], index=players, name="Rating")

    return ratings


def count_inversions_against_ground_truth(
    estimated_scores: List[float],
    ground_truth_scores: List[float],
) -> int:
    """
    Counts inversions between estimated Bradley-Terry ordering and ground-truth ordering.

    Higher score is considered better.

    Ties in either ordering are ignored.
    """

    n = len(ground_truth_scores)
    inversions = 0

    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = ground_truth_scores[i] - ground_truth_scores[j]
            est_diff = estimated_scores[i] - estimated_scores[j]

            if gt_diff == 0 or est_diff == 0:
                continue

            if gt_diff * est_diff < 0:
                inversions += 1

    return inversions


def load_existing_match_results(
    matches_save_file: str,
) -> Dict[Tuple[str, str, str], Dict]:
    """
    Loads already-computed match results.

    Returns:
        Mapping:
            (target_id, canonical_model_a, canonical_model_b) -> {
                "winner_id": winner candidate id,
                "winner_num": "1" or "2" relative to the stored CSV row if known,
            }

    Supports both old CSVs with:
        target_id,model_a,model_b,winner

    and newer CSVs with:
        target_id,model_a,model_b,winner,winner_id,round
    """

    existing = {}

    if not os.path.exists(matches_save_file):
        return existing

    with open(matches_save_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            target_id = str(row["target_id"])
            model_a = str(row["model_a"])
            model_b = str(row["model_b"])

            winner_raw = str(row["winner"])

            if "winner_id" in row and row["winner_id"] not in [None, ""]:
                winner_id = str(row["winner_id"])
            else:
                if winner_raw == "1":
                    winner_id = model_a
                elif winner_raw == "2":
                    winner_id = model_b
                else:
                    # If winner column already stores the winner id.
                    winner_id = winner_raw

            canonical_a, canonical_b = sorted((model_a, model_b))
            key = (target_id, canonical_a, canonical_b)

            existing[key] = {
                "winner_id": winner_id,
                "winner_num": winner_raw if winner_raw in ["1", "2"] else None,
            }

    return existing


def append_matches_to_csv(rows: List[Dict], matches_save_file: str) -> None:
    if not rows:
        return

    output_dir = os.path.dirname(matches_save_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(matches_save_file)

    fieldnames = [
        "target_id",
        "model_a",
        "model_b",
        "winner",
        "winner_id",
        "round",
    ]

    with open(matches_save_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def compute_checkpoint_inversions_for_target(
    target_id: str,
    candidate_ids: List[str],
    ground_truth_scores: List[float],
    match_results_for_bt: List[Dict],
    bt_C: float,
) -> Optional[int]:
    """
    Computes Bradley-Terry ratings from matches played so far for one target,
    then compares the resulting ordering with relevance_scores_full.
    """

    if len(match_results_for_bt) == 0:
        logger.warning(f"No match results for target {target_id}; skipping checkpoint.")
        return None

    results_df = pd.DataFrame(match_results_for_bt)

    try:
        ratings = compute_bt_ratings(
            results_df,
            C=bt_C,
            solver="lbfgs",
            max_iter=1000,
        )
    except Exception as e:
        logger.warning(
            f"Failed to compute Bradley-Terry ratings for target {target_id}: {repr(e)}"
        )
        return None

    estimated_scores = [
        float(ratings.get(candidate_id, 0.0)) for candidate_id in candidate_ids
    ]

    inversions = count_inversions_against_ground_truth(
        estimated_scores=estimated_scores,
        ground_truth_scores=ground_truth_scores,
    )

    return inversions


def run_swiss_tour_with_prompt_judge(
    targets: List[Dict],
    samples_all: List[List[Dict]],
    executor: QueryExecutor,
    existing_results: Dict[Tuple[str, str, str], Dict],
    num_rounds: int,
    checkpoint_every: int,
    num_players: int,
    bt_C: float,
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    Runs Swiss tournament using actual prompt executions.

    Ground-truth relevance_scores_full is used only for inversion evaluation.

    Returns:
        - dataframe of newly computed matches
        - mapping checkpoint_round -> list of inversion counts for targets in this chunk
    """

    target_ids = [str(target["id"]) for target in targets]

    scores = {}
    candidate_ids_by_target = {}
    ground_truth_by_target = {}
    pairs_played = {}
    match_results_for_bt = {}

    samples_dict = {
        str(sample["id"]): sample for samples in samples_all for sample in samples
    }

    for target, samples in zip(targets, samples_all):
        target_id = str(target["id"])

        candidate_ids = [str(sample["id"]) for sample in samples[:num_players]]
        candidate_ids_by_target[target_id] = candidate_ids

        scores[target_id] = {candidate_id: 0 for candidate_id in candidate_ids}
        pairs_played[target_id] = []
        match_results_for_bt[target_id] = []

        ground_truth_by_target[target_id] = [
            float(x) for x in target["relevance_scores_full"][:num_players]
        ]

    newly_computed_rows = []

    checkpoint_inversions = {
        r: [] for r in range(checkpoint_every, num_rounds + 1, checkpoint_every)
    }

    def apply_match_result(
        target_id: str,
        model_a: str,
        model_b: str,
        winner_id: str,
    ) -> None:
        if winner_id not in scores[target_id]:
            raise ValueError(
                f"Winner id {winner_id} is not a candidate for target {target_id}."
            )

        scores[target_id][winner_id] += 1

        match_results_for_bt[target_id].append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner_id,
            }
        )

    for round_idx in range(1, num_rounds + 1):
        print(f"======== Round {round_idx}/{num_rounds} ========")

        new_pairs = {}

        for target_id in target_ids:
            new_pairs[target_id] = next_pairs(
                scores=scores[target_id],
                pairs_played=pairs_played[target_id],
                num_players=num_players,
            )

            pairs_played[target_id].extend(new_pairs[target_id])

        batch_prompts = []
        cached_prompts = []

        for target in targets:
            target_id = str(target["id"])

            for s1, s2 in new_pairs[target_id]:
                s1 = str(s1)
                s2 = str(s2)

                canonical_s1, canonical_s2 = sorted((s1, s2))
                key = (target_id, canonical_s1, canonical_s2)

                prompt = {
                    "target_problem": target["problem"],
                    "target_solution": target["solution"],
                    "sample1_problem": samples_dict[s1]["problem"],
                    "sample1_solution": samples_dict[s1]["solution"],
                    "sample2_problem": samples_dict[s2]["problem"],
                    "sample2_solution": samples_dict[s2]["solution"],
                    "target_id": target_id,
                    "sample1_id": s1,
                    "sample2_id": s2,
                    "round": round_idx,
                }

                if key in existing_results:
                    cached_prompts.append(prompt)
                else:
                    batch_prompts.append(prompt)

        # Apply cached results without querying.
        for prompt in cached_prompts:
            target_id = prompt["target_id"]
            s1 = str(prompt["sample1_id"])
            s2 = str(prompt["sample2_id"])

            canonical_s1, canonical_s2 = sorted((s1, s2))
            key = (target_id, canonical_s1, canonical_s2)

            cached = existing_results[key]
            winner_id = str(cached["winner_id"])

            if winner_id not in [s1, s2]:
                logger.warning(
                    f"Cached winner {winner_id} is not in pair {(s1, s2)} "
                    f"for target {target_id}. Skipping cached result."
                )
                continue

            apply_match_result(
                target_id=target_id,
                model_a=s1,
                model_b=s2,
                winner_id=winner_id,
            )

        logger.info(
            f"Round {round_idx}: running {len(batch_prompts)} new queries; "
            f"using {len(cached_prompts)} cached results."
        )

        failed_prompts = []

        # Execute new prompts.
        if len(batch_prompts) > 0:
            for idx, messages, detailed_cost in executor.execute(batch_prompts):
                prompt = batch_prompts[idx]

                last_content = messages[-1]["content"]

                if isinstance(last_content, list):
                    last_content = last_content[-1]["content"]

                final_answer = fix_thinking(last_content)

                target_id = prompt["target_id"]
                s1 = str(prompt["sample1_id"])
                s2 = str(prompt["sample2_id"])

                try:
                    winner_num = extract_boxed_number(final_answer)

                    if winner_num == 1:
                        winner_id = s1
                    elif winner_num == 2:
                        winner_id = s2
                    else:
                        raise ValueError("No valid \\boxed{1} or \\boxed{2} found")

                    apply_match_result(
                        target_id=target_id,
                        model_a=s1,
                        model_b=s2,
                        winner_id=winner_id,
                    )

                    row = {
                        "target_id": target_id,
                        "model_a": s1,
                        "model_b": s2,
                        "winner": winner_num,
                        "winner_id": winner_id,
                        "round": round_idx,
                    }

                    newly_computed_rows.append(row)

                    canonical_s1, canonical_s2 = sorted((s1, s2))
                    existing_results[(target_id, canonical_s1, canonical_s2)] = {
                        "winner_id": winner_id,
                        "winner_num": str(winner_num),
                    }

                except ValueError:
                    logger.warning(
                        f"Parsing failed for target {target_id}, pair {(s1, s2)}. "
                        f"Queuing for retry."
                    )
                    failed_prompts.append(prompt)

        # Retry failed prompts once.
        if failed_prompts:
            logger.info(f"Retrying {len(failed_prompts)} failed queries...")

            for idx, messages, detailed_cost in executor.execute(failed_prompts):
                prompt = failed_prompts[idx]

                last_content = messages[-1]["content"]

                if isinstance(last_content, list):
                    last_content = last_content[-1]["content"]

                final_answer = fix_thinking(last_content)

                target_id = prompt["target_id"]
                s1 = str(prompt["sample1_id"])
                s2 = str(prompt["sample2_id"])

                try:
                    winner_num = extract_boxed_number(final_answer)

                    if winner_num == 1:
                        winner_id = s1
                    elif winner_num == 2:
                        winner_id = s2
                    else:
                        raise ValueError("No valid \\boxed{1} or \\boxed{2} found")

                    apply_match_result(
                        target_id=target_id,
                        model_a=s1,
                        model_b=s2,
                        winner_id=winner_id,
                    )

                    row = {
                        "target_id": target_id,
                        "model_a": s1,
                        "model_b": s2,
                        "winner": winner_num,
                        "winner_id": winner_id,
                        "round": round_idx,
                    }

                    newly_computed_rows.append(row)

                    canonical_s1, canonical_s2 = sorted((s1, s2))
                    existing_results[(target_id, canonical_s1, canonical_s2)] = {
                        "winner_id": winner_id,
                        "winner_num": str(winner_num),
                    }

                except ValueError:
                    logger.error(
                        f"Failed again for target {target_id}, pair {(s1, s2)}. "
                        f"Skipping entirely."
                    )
                    continue

        # Checkpoint: compute Bradley-Terry ratings and inversions.
        if round_idx % checkpoint_every == 0:
            logger.info(f"Computing checkpoint inversions after round {round_idx}.")

            for target_id in target_ids:
                inversions = compute_checkpoint_inversions_for_target(
                    target_id=target_id,
                    candidate_ids=candidate_ids_by_target[target_id],
                    ground_truth_scores=ground_truth_by_target[target_id],
                    match_results_for_bt=match_results_for_bt[target_id],
                    bt_C=bt_C,
                )

                if inversions is not None:
                    checkpoint_inversions[round_idx].append(inversions)

    df = pd.DataFrame(
        newly_computed_rows,
        columns=[
            "target_id",
            "model_a",
            "model_b",
            "winner",
            "winner_id",
            "round",
        ],
    )

    return df, checkpoint_inversions


def save_json(output: Dict, output_json: str) -> None:
    output_dir = os.path.dirname(output_json)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a Swiss tournament using actual LLM judge prompts, "
            "then compare Bradley-Terry rankings against relevance_scores_full."
        )
    )

    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--targets_dataset",
        default=None,
        help=(
            "HF dataset containing relevance_scores_full. "
            "If omitted, uses config['hf_datasets']['original_targets_dataset']."
        ),
    )

    parser.add_argument(
        "--candidates_dataset",
        default=None,
        help=(
            "HF candidates dataset. "
            "If omitted, uses config['hf_datasets']['original_candidates_dataset']."
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split. Default: train.",
    )

    parser.add_argument(
        "--output_json",
        required=True,
        help="Where to save average inversion results.",
    )

    parser.add_argument(
        "--matches_save_file",
        default=None,
        help=(
            "CSV file for saving judged matches. "
            "If omitted, uses config['swiss_tournament']['matches_save_file']."
        ),
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=NUM_ROUNDS_DEFAULT,
        help=f"Number of Swiss rounds. Default: {NUM_ROUNDS_DEFAULT}.",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
        help=f"Compute inversions every this many rounds. Default: {CHECKPOINT_EVERY_DEFAULT}.",
    )

    parser.add_argument(
        "--num_players",
        type=int,
        default=NUM_PLAYERS_DEFAULT,
        help=f"Number of candidates per target. Default: {NUM_PLAYERS_DEFAULT}.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Number of targets per chunk. Default: {BATCH_SIZE_DEFAULT}.",
    )

    parser.add_argument(
        "--bt_C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Bradley-Terry logistic regression.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--force_recalc",
        action="store_true",
        help="Ignore and overwrite existing match results.",
    )

    args = parser.parse_args()

    if args.num_rounds < 1:
        raise ValueError("--num_rounds must be positive.")

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint_every must be positive.")

    if args.num_players < 2:
        raise ValueError("--num_players must be at least 2.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prompt_file = config["prompts_pairwise_comparison"]["no-ties"]

    targets_dataset = (
        args.targets_dataset
        if args.targets_dataset is not None
        else config["hf_datasets"]["original_targets_dataset"]
    )

    candidates_dataset = (
        args.candidates_dataset
        if args.candidates_dataset is not None
        else config["hf_datasets"]["original_candidates_dataset"]
    )

    matches_save_file = (
        args.matches_save_file
        if args.matches_save_file is not None
        else config["swiss_tournament"]["matches_save_file"]
    )

    if args.force_recalc and os.path.exists(matches_save_file):
        logger.warning(f"--force_recalc passed. Removing {matches_save_file}.")
        os.remove(matches_save_file)

    with open(prompt_file, "r", encoding="utf-8") as file:
        template_string = file.read()

    api_cfg = config.get("judge_api", {})

    logger.info(f"Loading targets dataset: {targets_dataset}")
    logger.info(f"Loading candidates dataset: {candidates_dataset}")

    targets_ds = load_dataset(targets_dataset)[args.split]
    candidates_ds = load_dataset(candidates_dataset)[args.split]

    required_column = "relevance_scores_full"

    if required_column not in targets_ds.column_names:
        raise ValueError(
            f"Targets dataset must contain column '{required_column}'. "
            f"Available columns: {targets_ds.column_names}"
        )

    valid_targets = []
    skipped_missing_relevance = 0
    skipped_bad_candidates = 0

    logger.info("Filtering targets with valid relevance_scores_full...")

    for target in tqdm(targets_ds, desc="Filtering targets"):
        relevance_scores_full = target[required_column]

        if not has_valid_relevance_scores_full(
            relevance_scores_full=relevance_scores_full,
            num_players=args.num_players,
        ):
            skipped_missing_relevance += 1
            continue

        if "candidates" not in target or target["candidates"] is None:
            skipped_bad_candidates += 1
            continue

        if len(target["candidates"]) < args.num_players:
            skipped_bad_candidates += 1
            continue

        valid_targets.append(dict(target))

    if len(valid_targets) == 0:
        raise ValueError("No valid targets found with non-None relevance_scores_full.")

    logger.info(f"Total targets: {len(targets_ds)}")
    logger.info(f"Valid targets: {len(valid_targets)}")
    logger.info(
        f"Skipped missing/invalid relevance_scores_full: {skipped_missing_relevance}"
    )
    logger.info(f"Skipped bad candidates: {skipped_bad_candidates}")

    existing_results = {}

    if not args.force_recalc:
        existing_results = load_existing_match_results(matches_save_file)
        logger.info(f"Loaded {len(existing_results)} existing match results.")

    checkpoints = list(
        range(args.checkpoint_every, args.num_rounds + 1, args.checkpoint_every)
    )

    inversion_sums = {r: 0.0 for r in checkpoints}
    inversion_counts = {r: 0 for r in checkpoints}

    num_chunks = math.ceil(len(valid_targets) / args.batch_size)

    querier = APIQuery(
        model=api_cfg.get("model", "openai/gpt-oss-120b"),
        api=api_cfg.get("api", "vllm"),
        max_tokens=api_cfg.get("max_tokens", 50000),
        read_cost=api_cfg.get("read_cost", 0.15),
        write_cost=api_cfg.get("write_cost", 0.60),
        concurrent_requests=api_cfg.get("concurrent_requests", 16),
        reasoning_effort=api_cfg.get("reasoning_effort", "high"),
        human_readable_id=api_cfg.get(
            "human_readable_id",
            "GPT OSS 120B (high-2)",
        ),
        date=api_cfg.get("date", "2025-08-05"),
    )

    executor = QueryExecutor(
        querier=querier,
        prompt_template=template_string,
        system_prompt=api_cfg.get("system_prompt", "You are a helpful expert."),
    )

    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.batch_size
        end = min(start + args.batch_size, len(valid_targets))

        chunk = valid_targets[start:end]

        print(
            f"Starting valid target chunk {chunk_idx + 1}/{num_chunks}: "
            f"{start} - {end - 1}"
        )

        samples_all = []

        for target in chunk:
            samples = [
                dict(candidates_ds[int(candidate_idx)])
                for candidate_idx in target["candidates"][: args.num_players]
            ]
            samples_all.append(samples)

        df_new, checkpoint_inversions = run_swiss_tour_with_prompt_judge(
            targets=chunk,
            samples_all=samples_all,
            executor=executor,
            existing_results=existing_results,
            num_rounds=args.num_rounds,
            checkpoint_every=args.checkpoint_every,
            num_players=args.num_players,
            bt_C=args.bt_C,
        )

        new_rows = df_new.to_dict(orient="records")
        append_matches_to_csv(new_rows, matches_save_file)

        for r in checkpoints:
            invs = checkpoint_inversions.get(r, [])

            for inv in invs:
                inversion_sums[r] += inv
                inversion_counts[r] += 1

        average_inversions_so_far = {
            f"average_num_inversions_after_{r}_rounds": (
                inversion_sums[r] / inversion_counts[r]
                if inversion_counts[r] > 0
                else None
            )
            for r in checkpoints
        }

        output_so_far = {
            **average_inversions_so_far,
            "metadata": {
                "status": "partial" if chunk_idx + 1 < num_chunks else "complete",
                "config_file": args.config_file,
                "targets_dataset": targets_dataset,
                "candidates_dataset": candidates_dataset,
                "split": args.split,
                "matches_save_file": matches_save_file,
                "num_rounds": args.num_rounds,
                "checkpoint_every": args.checkpoint_every,
                "num_players": args.num_players,
                "batch_size": args.batch_size,
                "bt_C": args.bt_C,
                "seed": args.seed,
                "total_targets": len(targets_ds),
                "valid_targets": len(valid_targets),
                "processed_valid_targets": end,
                "skipped_missing_or_invalid_relevance_scores_full": skipped_missing_relevance,
                "skipped_bad_candidates": skipped_bad_candidates,
                "inversion_counts_by_checkpoint": inversion_counts,
            },
        }

        save_json(output_so_far, args.output_json)

        print(
            f"Finished valid target chunk {chunk_idx + 1}/{num_chunks}. "
            f"Intermediate results saved to {args.output_json}."
        )

    average_inversions = {
        f"average_num_inversions_after_{r}_rounds": (
            inversion_sums[r] / inversion_counts[r] if inversion_counts[r] > 0 else None
        )
        for r in checkpoints
    }

    final_output = {
        **average_inversions,
        "metadata": {
            "status": "complete",
            "config_file": args.config_file,
            "targets_dataset": targets_dataset,
            "candidates_dataset": candidates_dataset,
            "split": args.split,
            "matches_save_file": matches_save_file,
            "num_rounds": args.num_rounds,
            "checkpoint_every": args.checkpoint_every,
            "num_players": args.num_players,
            "batch_size": args.batch_size,
            "bt_C": args.bt_C,
            "seed": args.seed,
            "total_targets": len(targets_ds),
            "valid_targets": len(valid_targets),
            "skipped_missing_or_invalid_relevance_scores_full": skipped_missing_relevance,
            "skipped_bad_candidates": skipped_bad_candidates,
            "inversion_counts_by_checkpoint": inversion_counts,
        },
    }

    save_json(final_output, args.output_json)

    print(f"Saved final results to {args.output_json}")
    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
