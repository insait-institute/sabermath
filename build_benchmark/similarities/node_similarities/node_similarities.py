import argparse
import json
import itertools

import numpy as np
from tqdm import tqdm

from metrics import lin_similarity


def count_nodes(tree: dict) -> int:
    count = 0

    def _rec(node):
        nonlocal count
        count += 1

        for child in node.get("links", []):
            _rec(child)

    _rec(tree)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tree_path",
        type=str,
        help="Path to the frequency tag tree",
    )

    parser.add_argument(
        "npy_output",
        type=str,
        help="Path to where the ouput numpy similarity matrix will be saved",
    )

    args = parser.parse_args()

    print("[~] Loading tree..")
    with open(args.tree_path, "r") as file:
        tree = json.load(file)

    node_count = count_nodes(tree)

    similarities = np.zeros((node_count, node_count))

    def iter_nodes(node):
        yield node
        for child in node.get("links", []):
            yield from iter_nodes(child)

    def compute_sim_rec(node, pbar):
        nonlocal similarities

        prob_mica = node["frequency"]
        idx_mica = node["idx"]

        children = node.get("links", [])
        pairs = list(itertools.combinations(children, 2))

        for n in iter_nodes(node):
            idx = n["idx"]
            prob = n["frequency"]
            sim = lin_similarity(prob, prob_mica, prob_mica)
            similarities[idx_mica][idx] = sim
            similarities[idx][idx_mica] = sim
            pbar.update(2)

        for n1, n2 in pairs:
            for d1, d2 in itertools.product(iter_nodes(n1), iter_nodes(n2)):
                idx1, idx2 = d1["idx"], d2["idx"]
                prob1, prob2 = d1["frequency"], d2["frequency"]
                sim = lin_similarity(prob1, prob2, prob_mica)
                similarities[idx1][idx2] = sim
                similarities[idx2][idx1] = sim
                pbar.update(2)

        for child in children:
            compute_sim_rec(child, pbar)

    np.fill_diagonal(similarities, 1.0)

    # for progress bar - all but the diagonal pairs
    total = node_count**2 - node_count

    with tqdm(total=total, desc="Computing similarities...") as pbar:
        compute_sim_rec(tree, pbar)
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)

    print("[~] Saving...")
    np.save(args.npy_output, similarities)
    print("[+] All done.")


if __name__ == "__main__":
    main()
