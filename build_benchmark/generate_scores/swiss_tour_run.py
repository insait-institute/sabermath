import random
import re
import os
from loguru import logger
from typing import List, Tuple, Dict
from datasets import load_dataset
import pandas as pd
import argparse
import csv
import yaml

from llmteach.executor import QueryExecutor
from llmteach.api import APIQuery
from llmteach.postprocess import fix_thinking


def extract_boxed_number(s: str) -> int:
    match = re.search(r"\\boxed\{\s*([12])\s*\}", s)
    if not match:
        raise ValueError("No valid \\boxed{1} or \\boxed{2} found")
    return int(match.group(1))


def next_pairs(
    scores: Dict[str, int], pairs_played: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Generates the next n/2 pairs, pairing based on current scores and without repeating matches
    """

    items = list(scores.keys())
    random.shuffle(
        items
    )  # random shuffle, specifically important in the beginning, in the next steps it ensures shuffle of same values
    ids_sorted = sorted(items, key=lambda x: scores[x])

    begin, end = 0, 150

    used = set()
    pairings = []

    idxs_not_paired = []

    for i, player in enumerate(ids_sorted):

        if i < begin or i > end:
            continue

        if player in used:
            continue

        for opponent in ids_sorted[i + 1 : end]:

            if opponent in used:
                continue

            new_pair = (player, opponent)
            new_pair_r = (opponent, player)

            if new_pair in pairs_played or new_pair_r in pairs_played:
                continue

            pairings.append(new_pair)
            used.add(player)
            used.add(opponent)

            break

        if not player in used:
            idxs_not_paired.append(i)

    not_used = set(ids_sorted[begin:end]).difference(used)

    if len(not_used) != 0:
        print(
            f"WARNING: {len(not_used)} people not playing: {[x for x in idxs_not_paired]}"
        )

    return pairings


def sort_pairs(pairings: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_pairs = [tuple(sorted(x)) for x in pairings]
    return new_pairs


def run_swiss_tour(
    targets: List[Dict], samples_all: List[List[Dict]], calculated: List[Dict]
):

    target_ids = [target["id"] for target in targets]

    scores = {
        target_id: {x: 0 for x in [sample["id"] for sample in samples]}
        for (target_id, samples) in zip(target_ids, samples_all)
    }  # target : sample_id : score dict

    samples_dict = {
        sample["id"]: sample
        for sample in [item for sublist in samples_all for item in sublist]
    }

    pairs_played = {target: [] for target in target_ids}
    match_results = []

    for r in range(num_rounds):

        print(f"========Round {r}/{num_rounds}========")

        new_pairs = {}

        for target_id in target_ids:

            new_pairs[target_id] = next_pairs(
                scores[target_id], pairs_played[target_id]
            )
            new_pairs[target_id] = sort_pairs(new_pairs[target_id])
            pairs_played[target_id].extend(new_pairs[target_id])

        batch_prompts = [
            {
                "target_problem": target["problem"],
                "target_solution": target["solution"],
                "sample1_problem": samples_dict[s1]["problem"],
                "sample1_solution": samples_dict[s1]["solution"],
                "sample2_problem": samples_dict[s2]["problem"],
                "sample2_solution": samples_dict[s2]["solution"],
                "target_id": target["id"],
                "sample1_id": s1,
                "sample2_id": s2,
            }
            for target in targets
            for (s1, s2) in new_pairs[target["id"]]
        ]

        calculated_set = {
            (d["target_id"], d["sample1_id"], d["sample2_id"]) for d in calculated
        }

        filtered_batch = [
            d
            for d in batch_prompts
            if (d["target_id"], d["sample1_id"], d["sample2_id"]) not in calculated_set
        ]

        logger.info(
            f"Running batch with {len(filtered_batch)} instead of {len(batch_prompts)} queries..."
        )

        failed_prompts = []

        # Execute Batch
        for idx, messages, detailed_cost in executor.execute(filtered_batch):

            prompt = filtered_batch[idx]

            last_content = messages[-1]["content"]

            if isinstance(last_content, list):
                last_content = last_content[-1]["content"]

            final_answer = fix_thinking(last_content)

            curr_target_id = prompt["target_id"]
            id_pair = (prompt["sample1_id"], prompt["sample2_id"])

            try:

                winner = extract_boxed_number(final_answer)

                if winner == 1:
                    scores[curr_target_id][id_pair[0]] += 1
                elif winner == 2:
                    scores[curr_target_id][id_pair[1]] += 1
                else:
                    raise ValueError("No valid \\boxed{1} or \\boxed{2} found")

                match_results.append(
                    {
                        "target_id": curr_target_id,
                        "model_a": id_pair[0],
                        "model_b": id_pair[1],
                        "winner": winner,
                    }
                )

            except ValueError:
                logger.warning(f"Parsing failed for pair {id_pair}. Queuing for retry.")
                failed_prompts.append(filtered_batch[idx])

        if failed_prompts:
            logger.info(f"Retrying {len(failed_prompts)} failed queries...")

            for idx, messages, detailed_cost in executor.execute(failed_prompts):

                prompt = failed_prompts[idx]

                last_content = messages[-1]["content"]

                if isinstance(last_content, list):
                    last_content = last_content[-1]["content"]

                final_answer = fix_thinking(last_content)

                curr_target_id = prompt["target_id"]
                id_pair = (prompt["sample1_id"], prompt["sample2_id"])

                try:
                    winner = extract_boxed_number(final_answer)
                    if winner == 1:
                        scores[curr_target_id][id_pair[0]] += 1
                    elif winner == 2:
                        scores[curr_target_id][id_pair[1]] += 1
                    else:
                        raise ValueError("No valid \\boxed{1} or \\boxed{2} found")

                    match_results.append(
                        {
                            "target_id": curr_target_id,
                            "model_a": id_pair[0],
                            "model_b": id_pair[1],
                            "winner": winner,
                        }
                    )

                except ValueError:
                    # If it fails the second time, we just skip it
                    logger.error(f"Failed again for pair {id_pair}. Skipping entirely.")
                    continue

    df = pd.DataFrame(
        match_results, columns=["target_id", "model_a", "model_b", "winner"]
    )
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run prompts for a list of target indices."
    )

    parser.add_argument("--config_file", required=True)

    parser.add_argument(
        "--force_recalc",
        action="store_true",
        help="Recalculate prompts even if results already exist",
    )

    args = parser.parse_args()
    FORCE_RECALCULATE = args.force_recalc

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    num_rounds = config["swiss_tournament"]["num_rounds"]
    prompt_file = config["prompts_pairwise_comparison"]["no-ties"]
    save_file_name = config["swiss_tournament"]["matches_save_file"]

    targets_dataset = config["hf_datasets"]["original_targets_dataset"]
    candidates_dataset = config["hf_datasets"]["original_candidates_dataset"]

    with open(prompt_file, "r", encoding="utf-8") as file:
        template_string = file.read()

    querier = APIQuery(
        model="openai/gpt-oss-120b",
        api="vllm",
        max_tokens=50000,
        read_cost=0.15,
        write_cost=0.60,
        concurrent_requests=16,
        reasoning_effort="high",
        human_readable_id="GPT OSS 120B (high-2)",
        date="2025-08-05",
    )

    executor = QueryExecutor(
        querier=querier,
        prompt_template=template_string,
        system_prompt="You are a helpful expert.",  # Optional
    )

    targets = load_dataset(targets_dataset)["train"]
    candidates = load_dataset(candidates_dataset)["train"]

    calculated = []
    if os.path.exists(save_file_name) and not FORCE_RECALCULATE:
        with open(save_file_name, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                targ, cand_1, cand_2, _ = line
                calculated.append(
                    {"target_id": targ, "sample1_id": cand_1, "sample2_id": cand_2}
                )

    batch_size = 200

    for i in range(200, len(targets), batch_size):

        chunk = targets.select(range(i, min(i + batch_size, len(targets))))

        print(f"Starting target {i} - {i+batch_size-1}/{len(targets)}")

        cands_target = [
            [candidates[x] for x in target["candidates"]] for target in chunk
        ]

        write_header = not os.path.exists(save_file_name)
        df = run_swiss_tour(chunk, cands_target, calculated)
        df.to_csv(save_file_name, mode="a", index=False, header=write_header)

        print(
            f"Finished target {i} - {i+batch_size-1}/{len(targets)}, results saved to {save_file_name}"
        )
