import argparse
import csv
from datasets import load_dataset, Dataset
import pandas as pd
import tqdm
import yaml

from bradley_terry import compute_bt_ratings

MAX_SCORE_VAL = 1


def matches_dict_from_file(matches_res_file: str):

    matches_dict = {}

    with open(matches_res_file) as f:
        reader = csv.reader(f)

        row = next(reader)
        for row in reader:
            target, id1, id2, win = row
            winner = id1 if win == "1" else id2

            if not matches_dict.get(target, False):
                matches_dict[target] = {"model_a": [], "model_b": [], "winner": []}

            matches_dict[target]["model_a"].append(id1)
            matches_dict[target]["model_b"].append(id2)
            matches_dict[target]["winner"].append(winner)

    return matches_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument(
        "--mode",
        choices=["SWISS", "FULL"],
        default="SWISS",
    )
    parser.add_argument("--idxs", type=int, nargs="+")

    args = parser.parse_args()

    config_file = args.config_file
    mode = args.mode
    idxs = args.idxs

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    targets_dataset = config["hf_datasets"]["original_targets_dataset"]
    candidates_dataset = config["hf_datasets"]["original_candidates_dataset"]

    output_dataset = config["hf_datasets"]["targets_with_relevances_dataset"]

    targets = load_dataset(targets_dataset)["train"]
    candidates = load_dataset(candidates_dataset)["train"]

    if (
        idxs is None
    ):  # in case the idxs argument is not passed, relevances will be computed for all targets
        idxs = range(len(targets))

    cand_id_dict = {candidates[i]["id"]: i for i in range(len(candidates))}

    targets_rel = []

    if mode == "SWISS":
        matches_dict = matches_dict_from_file(
            config["swiss_tournament"]["matches_save_file"]
        )

    for i in idxs:

        target = targets[i]

        if mode == "FULL":
            matches_dict = matches_dict_from_file(
                f"calculated_csvs/matches_all_{i}.csv"
            )

        target_id = target["id"]
        results_pd = pd.DataFrame(matches_dict[target_id])
        ratings, _ = compute_bt_ratings(results_pd)
        candidates_ids = ratings.keys().tolist()

        result_list = []
        for _ in range(150):
            c_id = candidates[target["candidates"][_]]["id"]
            result_list.append(ratings[c_id].item())

        min_val = min(result_list)
        max_val = max(result_list)
        scaled = [
            (x - min_val) / (max_val - min_val) * MAX_SCORE_VAL for x in result_list
        ]

        if mode == "SWISS":
            target["relevance_scores"] = scaled
        elif mode == "FULL":
            target["relevance_scores_full"] = scaled

        targets_rel.append(target)

    dataset = Dataset.from_list(targets_rel)
    dataset.push_to_hub(output_dataset)
