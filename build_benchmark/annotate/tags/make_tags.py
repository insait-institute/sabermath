import argparse
import asyncio

from datasets import load_dataset

from tree import load_tag_tree
from tagger import TagMaker


def _idx_map(lens: list[int]) -> list[int]:
    def make_f():
        count = 0

        def f(x: int):
            nonlocal count
            count += x
            return count

        return f

    f = make_f()
    return list(map(f, [0, *lens]))


async def main() -> None:
    parser = argparse.ArgumentParser(prog="make_tags.py")

    parser.add_argument("hf_path", type=str, help="HF Path to the dataset to annotate")

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        required=False,
        help="Comma-separated list of datasets withing to path to annotate",
    )

    parser.add_argument(
        "--columns",
        type=str,
        default="problem,solution",
        required=False,
        help="List of columns to include for annotation",
    )

    parser.add_argument(
        "--out", type=str, default=None, required=False, help="Output HF path"
    )

    parser.add_argument(
        "--out-column",
        type=str,
        default="tags",
        required=False,
        help="Output HF Column",
    )

    parser.add_argument(
        "--tree",
        type=str,
        default="../../data/tree.json",
        required=False,
        help="Location of the JSON tag tree",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=False,
        help="OpenAI model to use for annotation",
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        default="high",
        required=False,
        help="Reasoning effort for the model",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.openai.com/v1",
        required=False,
        help="OpenAI API URL",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        required=False,
        help="Threshold for tag to include (0.0-1.0)",
    )

    parser.add_argument(
        "--silent", action="store_true", help="Prevent info logging when running"
    )

    args = parser.parse_args()

    verbose = not args.silent

    if args.threshold < 0.0 or args.threshold > 1.0:
        print("[ERROR] --threshold must be a float in [0.0, 1.0]")
        return

    if verbose:
        print("[~] Loading tree...")
    try:
        tree = load_tag_tree(args.tree)
        if verbose:
            print("[+] Tree loaded")
    except Exception as e:
        print(f"[ERROR] Loading tree failed due to error: {e}")
        return

    columns = args.columns.split(",")

    if verbose:
        print("[~] Loading corpus...")
    try:
        ds = load_dataset(args.hf_path)
    except Exception as e:
        print(f"[ERROR] Loading '{args.hf_path}' failed due to error: {e}")
        return

    # if --datasets is not set, tag all datasets
    if args.datasets is None:
        datasets = list(ds.keys())
    else:
        datasets = args.datasets.split(",")

    for name in datasets:
        if name not in ds:
            print(f"[ERROR] No dataset '{name}' in '{args.hf_path}")
            return
        for col in columns:
            if col not in ds[name].column_names:
                print(f"[ERROR] No column '{col}' in dataset '{args.hf_path}:{name}'")
                return

    if verbose:
        print("[+] Corpus loaded.")

    tagger = TagMaker(
        tree,
        model=args.model,
        threshold=args.threshold,
        api_url=args.api_url,
        reasoning=args.reasoning,
    )

    texts = [
        "\n\n".join(f"{col.upper()}: {row[col]}" for col in columns)
        for name in datasets
        for row in ds[name]
    ]

    # boundry indices for each dataset
    idxs = _idx_map(len(ds[name]) for name in datasets)

    if verbose:
        print("[~] Annotating...")

    # push all texts to tagger, then split to datasets
    tags = await tagger.annotate(texts, show_progress_bar=verbose)

    for i, name in enumerate(datasets):
        begin, end = idxs[i], idxs[i + 1]
        ds[name] = ds[name].add_column(args.out_column, tags[begin:end])

    out_path = f"{args.hf_path}_tagged" if args.out is None else args.out

    if verbose:
        print("[+] Annotation finished.")
        print(f"[~] Uploading results to '{out_path}'...")

    ds.push_to_hub(out_path)

    if verbose:
        print("[+] Finished.")


if __name__ == "__main__":
    asyncio.run(main())
