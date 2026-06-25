import argparse
import random
import time

import numpy as np

# make sure to run
# pip install ..
import fastbma


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--similarities",
        type=str,
        default=None,
        required=False,
        help="Location of .npy file containing a similarity matrix",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        required=False,
        help="Number of pairwise index vectors to generate",
    )

    parser.add_argument(
        "--seed", type=int, default=None, required=False, help="Random generator seed"
    )

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        print(f"[+] Random seed set to {args.seed}.")
    else:
        print("[WARNING] Random seed not specified. Using default settings.")

    size = 100

    if args.similarities:
        print("[~] Loading Similarity Matrix...")
        sim_npy = np.load(args.similarities)
        size = sim_npy.shape[0]
        sim = fastbma.SimilarityMatrix(sim_npy)
        print("[+] Similarity Matrix loaded.")
    else:
        print("[~] No path to Similarity Matrix specified. Generating...")
        if args.seed:
            randgen = np.random.default_rng(seed=args.seed)
            sim_npy = randgen.random((size, size))
        else:
            sim_npy = np.random.random((size, size))
        sim = fastbma.SimilarityMatrix(sim_npy)
        print(f"[+] Generated a random matrix of shape ({size}, {size}).")

    print("[~] Generating index lists...")
    idxs = [[random.randrange(size) for _ in range(8)] for _ in range(args.count)]
    print("[+] Lists generated.")

    print("[~] Starting BMA evaluation.")

    start = time.perf_counter()
    result = fastbma.compute(sim, idxs, idxs)
    end = time.perf_counter()

    elapsed = end - start
    total = args.count**2
    ns_per_iter = (elapsed * 1e9) / total
    iter_per_s = total / elapsed

    print(f"[+] Finished computing {total} BMA scores.")
    print(f"Computation finished in\t{elapsed:.2f} s")
    print(f"Time per BMA score:\t{ns_per_iter:.2f} ns")
    print(f"BMA computation speed:\t{iter_per_s:.2f} scores/s")
    print("[+] All done.")


if __name__ == "__main__":
    main()
