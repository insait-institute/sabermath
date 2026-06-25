import argparse
from typing import Iterable

from datasets import (
    load_dataset,
    concatenate_datasets,
)
import numpy as np
from tqdm import tqdm

from similarities import get_similarities
from utils import load_data, get_domains, DOMAIN_LOWER_TO_STYLIZED


def get_targets_and_candidates(
    idxs: list[int],
    bma_threshold: float,
    jaccard_threshold: float,
    per_group_count: int,
    total_rows: int,
    *,
    path: str = "../data/sim/",
    skip_target_check: bool = False,
    output_candidates: bool = False,
    show_progress_bar: bool = True,
) -> tuple[list[int], list[list[int]] | None]:
    selected_idxs = []
    candidates: list[list[int]] = []
    rng = np.random.default_rng()

    for idx in tqdm(idxs, disable=not show_progress_bar):
        bma, jaccard = get_similarities(idx, total_rows, path)

        high_bma = bma >= bma_threshold
        high_jaccard = jaccard >= jaccard_threshold

        high_bma[idx] = False
        high_jaccard[idx] = False

        high_bma_low_jaccard = high_bma & ~high_jaccard
        low_bma_high_jaccard = ~high_bma & high_jaccard
        high_bma_high_jaccard = high_bma & high_jaccard

        if not skip_target_check:
            if np.sum(high_bma_low_jaccard) < per_group_count:
                continue
            if np.sum(low_bma_high_jaccard) < per_group_count:
                continue
            if np.sum(high_bma_high_jaccard) < per_group_count:
                continue

        selected_idxs.append(idx)

        if output_candidates:
            bma_candidates = np.flatnonzero(high_bma_low_jaccard)
            jaccard_candidates = np.flatnonzero(low_bma_high_jaccard)
            both_candidates = np.flatnonzero(high_bma_high_jaccard)

            b_scnt = min(per_group_count, len(bma_candidates))
            j_scnt = min(per_group_count, len(jaccard_candidates))
            both_scnt = min(per_group_count, len(both_candidates))

            bma_candidates = rng.choice(bma_candidates, size=b_scnt, replace=False)
            jaccard_candidates = rng.choice(
                jaccard_candidates, size=j_scnt, replace=False
            )
            both_candidates = rng.choice(both_candidates, size=both_scnt, replace=False)

            local_candidates = (
                bma_candidates.tolist()
                + jaccard_candidates.tolist()
                + both_candidates.tolist()
            )
            candidates.append(local_candidates)

    if output_candidates:
        return selected_idxs, candidates

    return selected_idxs, None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("databank", type=str, help="Source Databank")
    parser.add_argument("--bma", type=float, required=True)
    parser.add_argument("--jaccard", type=float, required=True)
    parser.add_argument("--per-group-count", type=int, default=50)
    parser.add_argument("--similarities", type=str, default="../data/sim/")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--private", action="store_true")

    parser.add_argument(
        "--domain",
        type=str.lower,
        choices=list(DOMAIN_LOWER_TO_STYLIZED.keys()) + ["all"],
        default="all",
        required=False,
        help="Limit targets to a domain",
    )

    args = parser.parse_args()

    if args.private and args.out is None:
        print("[-] To use '--private', you must set the '--out' parameter.")
        return

    path = args.similarities

    print("[~] Loading databank...")
    ds = load_data(args.databank)

    if args.domain == "all":
        domain = "Not specified."
        init_targets_idxs = list(range(ds.num_rows))
    else:
        domain = DOMAIN_LOWER_TO_STYLIZED.get(args.domain)
        print(f"[~] Filtering problems with domain: {domain}.")
        init_targets_idxs = [
            i for i in tqdm(range(ds.num_rows)) if domain in get_domains(ds, i)
        ]
        # init_targets_idxs = get_domain_idxs(ds, domain)

    print("[~] Selecting viable targets and generating candidates...")

    # OPTIMIZATION: Combine selection and sampling into one pass
    selected_targets, candidates = get_targets_and_candidates(
        init_targets_idxs,
        args.bma,
        args.jaccard,
        args.per_group_count,
        ds.num_rows,
        path=path,
        skip_target_check=False,
        output_candidates=True,
    )

    print(f"[+] Number of viable targets: {len(selected_targets)}.")

    if args.out is None:
        return

    print("[~] Creating an output dataset...")

    # FIX: Correct Indexing Logic
    # 1. Get all unique candidate IDs and sort them
    unique_candidate_idxs = sorted(list(set(idx for lst in candidates for idx in lst)))

    # 2. Map original index -> its position in the NEW candidates dataset
    idx_lookup_table = {old_idx: i for i, old_idx in enumerate(unique_candidate_idxs)}

    # 3. Create datasets using the sorted unique indices
    ds = ds.add_column("original_index", list(range(ds.num_rows)))
    targets_dataset = ds.select(selected_targets)
    candidates_dataset = ds.select(unique_candidate_idxs)

    # 4. Map candidates list to the NEW local indices
    candidates_column = [
        [idx_lookup_table[idx] for idx in candidate_lst] for candidate_lst in candidates
    ]

    # Retrieval for scores remains the same but uses the already generated candidates
    bma_column = []
    jaccard_column = []

    print("[~] Retrieving relevant scores...")
    for target, candidate_lst in tqdm(
        zip(selected_targets, candidates), total=len(selected_targets)
    ):
        bma_scores, jaccard_scores = get_similarities(target, ds.num_rows, path)
        bma_column.append([bma_scores[i] for i in candidate_lst])
        jaccard_column.append([jaccard_scores[i] for i in candidate_lst])

    targets_dataset = targets_dataset.add_column("candidates", candidates_column)
    targets_dataset = targets_dataset.add_column("bma_scores", bma_column)
    targets_dataset = targets_dataset.add_column("jaccard_scores", jaccard_column)

    print("[~] Pushing output to HuggingFace...")
    targets_dataset.push_to_hub(f"{args.out}_targets", private=args.private)
    candidates_dataset.push_to_hub(f"{args.out}_candidates", private=args.private)

    print("[+] All done.")


if __name__ == "__main__":
    main()
