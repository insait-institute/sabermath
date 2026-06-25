import argparse
import random

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# IMPORTANT: Each tuple must be sorted in alphabetic order
DOMAIN_DISTRIBUTION = {
    ("Algebra",): 150,
    ("Geometry",): 195,
    ("Number Theory",): 150,
    ("Calculus and Analysis",): 199,
    ("Combinatorics",): 150,
    ("Combinatorics", "Number Theory"): 50,
    ("Algebra", "Combinatorics"): 50,
    ("Algebra", "Number Theory"): 50,
    ("Algebra", "Geometry"): 3,
    ("Combinatorics", "Geometry"): 2,
    ("Algebra", "Calculus and Analysis"): 1,
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "targets",
        type=str,
        help="HF path to original targets dataset",
    )

    parser.add_argument(
        "candidates", type=str, help="HF path to original candidates dataset"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, it is used as a seed for sampling",
    )

    parser.add_argument(
        "--private", action="store_true", help="Store the output datasets a private"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[+] Set seed to: {args.seed}.")

    print("[+] Loading data...")
    targets = load_dataset(args.targets)["train"]
    candidates = load_dataset(args.candidates)["train"]

    buckets_original = {key: [] for key in DOMAIN_DISTRIBUTION.keys()}

    for i, row in enumerate(targets):
        domain = tuple(sorted(row["domains"]))
        if domain in buckets_original:
            buckets_original[domain].append(i)

    partial_datasets = []

    print("[~] Selecting problems...")

    for domains, idxs in tqdm(buckets_original.items()):
        count = DOMAIN_DISTRIBUTION[domains]
        if len(idxs) < count:
            raise RuntimeError(f"Number of problem of domain {domains} is too small.")
        selected = random.sample(idxs, count)
        ds_local = targets.select(selected)
        partial_datasets.append(ds_local)

    new_targets = concatenate_datasets(partial_datasets)

    # This part deletes unused candidate rows and remaps
    # idxs in the target dataset columns "candidates"

    print("[~] Removing unused candidate rows and remap indices in targets dataset...")

    used_old_indices = sorted({idx for row in new_targets["candidates"] for idx in row})

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_old_indices)}

    candidates_filtered = candidates.select(used_old_indices)

    def remap_indices(row):
        row["candidates"] = [old_to_new[idx] for idx in row["candidates"]]
        return row

    new_targets = new_targets.map(remap_indices)

    print("[~] Pushing output to HuggingFace...")
    new_targets.push_to_hub(f"{args.targets}_reduced", private=args.private)
    candidates_filtered.push_to_hub(f"{args.candidates}_reduced", private=args.private)

    print("[+] All done.")


if __name__ == "__main__":
    main()
