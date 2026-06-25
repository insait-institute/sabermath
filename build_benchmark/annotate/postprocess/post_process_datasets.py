import argparse
from datasets import load_dataset


def clear_tags(tags: list[str]) -> list[str]:
    tags = sorted(set(tags))
    result = []

    prev = None
    for t in tags:
        if prev is None:
            prev = t
            continue

        if t.startswith(prev.rstrip("/") + "/"):
            prev = t
        else:
            result.append(prev)
            prev = t

    if prev is not None:
        result.append(prev)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tags_dataset", type=str, help="HuggingFace path of the tags dataset"
    )

    parser.add_argument(
        "ideas_dataset", type=str, help="HuggingFace path of the ideas dataset"
    )

    parser.add_argument(
        "--tags-column",
        type=str,
        default="tags",
        required=False,
        help="Name of the column holding the tags",
    )

    parser.add_argument(
        "--idea-column",
        type=str,
        default="idea",
        required=False,
        help="Name of the column holding the core idea",
    )

    parser.add_argument("out_dataset", type=str, help="Output HuggingFace path")

    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload the output dataset as private",
    )

    args = parser.parse_args()

    tags_ds = load_dataset(args.tags_dataset)
    ideas_ds = load_dataset(args.ideas_dataset)

    datasets = tags_ds.keys()

    # begin validation
    if set(datasets) != set(ideas_ds.keys()):
        print("[ERROR] Incompatible input datasets")
        return

    for ds_name in datasets:
        if tags_ds[ds_name].num_rows != ideas_ds[ds_name].num_rows:
            print(f'[ERROR] Mismatch in lenght of datasets named "{ds_name}".')
            return
        if args.tags_column not in tags_ds[ds_name].column_names:
            print(
                f'[ERROR] No column "{args.tags_column}" in "{args.tags_dataset}:{ds_name}".'
            )
            return
        if args.idea_column not in ideas_ds[ds_name].column_names:
            print(
                f'[ERROR] No column "{args.idea_column}" in "{args.ideas_dataset}:{ds_name}".'
            )
            return
    # end validation

    for name in datasets:
        print(f'[~] Processing dataset "{name}"...')

        new_column = ideas_ds[name][args.idea_column]
        tags_ds[name] = tags_ds[name].add_column(args.idea_column, new_column)

        tags_ds[name] = tags_ds[name].map(
            lambda row: {args.tags_column: clear_tags(row[args.tags_column])}
        )

        tags_ds[name] = tags_ds[name].filter(
            lambda row: len(row[args.tags_column]) > 0
            and len(row[args.idea_column].strip()) > 0
        )

    print("[+] Processing finished. Uploading.")
    tags_ds.push_to_hub(args.out_dataset, private=args.private)
    print("[+] All done.")


if __name__ == "__main__":
    main()
