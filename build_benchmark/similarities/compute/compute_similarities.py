import argparse

import numpy as np

from tree import load_tree
from processor import SimilarityProcessor


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "hf_path",
        type=str,
        help="HuggingFace path of the annotated data bank",
    )

    parser.add_argument(
        "--tree",
        type=str,
        default="../../data/freq_tree.json",
        required=False,
        help="Path to the locally saved Tags Tree",
    )

    parser.add_argument(
        "--similarities",
        type=str,
        default="../../data/tree_lin_sim.npy",
        required=False,
        help="Path to the locally saved pairwise tree node similarities",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Don't distribute the workload across available CPUs",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        required=False,
        help="Maximum number of CPU cores to utilize",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="../../data/sim",
        required=False,
        help="Path there to save the similarity chunks",
    )

    parser.add_argument(
        "--chunk-count",
        type=int,
        default=100,
        required=False,
        help="Number of chunks to split the results into",
    )

    args = parser.parse_args()

    if args.no_parallel and args.max_parallel is not None:
        print('[-] Cannot set "--max-parallel" if "--no-parallel" is set.')
        return

    print("[~] Loading tree...")
    tree = load_tree(args.tree)

    print("[~] Loading node similarities...")
    node_sim = np.load(args.similarities)

    processor = SimilarityProcessor(
        args.hf_path, tree, node_sim, parallelize=(not args.no_parallel)
    )

    print("[+] Setup completed.")

    print("[~] Computing similarities...")
    processor.compute(args.out, args.chunk_count, max_parallel=args.max_parallel)

    print("[+] All done.")


if __name__ == "__main__":
    main()
