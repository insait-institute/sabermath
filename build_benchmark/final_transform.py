import argparse
from datasets import load_dataset
import numpy as np


def process(example):
    new_scores = np.array(example["relevance_scores"]) * 5
    example["relevance_scores"] = new_scores
    return example


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=str,
        help="HF Path to the dataset containing the raw scores",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        required=False,
        help="Output HF Path",
    )

    parser.add_argument(
        "--private", action="store_true", help="Export dataset as private"
    )

    args = parser.parse_args()

    print("[~] Loading model...")
    ds = load_dataset(args.dataset)["train"]

    ds = ds.map(process, desc="Preprocessing dataset")

    out = f"{args.dataset}_transformed_x5" if args.out is None else args.out

    print("[~] Saving to HuggingFace...")

    ds.push_to_hub(out, private=args.private)

    print("[+] All done.")


if __name__ == "__main__":
    main()
