"""
This script constructs a tree from the anotated
tags in the databank, calculating and assigning
frequency labels for each node.
"""

import argparse
import json

from datasets import load_dataset
from tqdm import tqdm


class Tree:
    _FREQ_LABEL = "frequency"
    _COUNT_LABEL = "_count"
    _IDX_LABEL = "idx"

    def __init__(self):
        self._tree = {}
        self._total_count = 0
        self._num_nodes = 0
        self._finalized = False

    def _isnode(self, node: dict, title: str):
        return "title" in node and node["title"] == title

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    def get_tree(self):
        if not self._finalized:
            raise RuntimeError("Tree not finalized.")
        if (
            "links" not in self._tree
            or not isinstance(self._tree["links"], list)
            or len(self._tree["links"]) == 0
        ):
            return {}
        if len(self._tree["links"]) != 1:
            raise RuntimeError("Tree has more than 1 root")
        return self._tree["links"][0]

    def save(self, path: str):
        tree = self.get_tree()
        with open(path, "w") as f:
            json.dump(tree, f, indent=2)

    def increment(self, tag_path: str) -> None:
        if self._finalized:
            raise RuntimeError("This tree structure has been finalized.")

        self._total_count += 1
        node = self._tree
        blocks = [b for b in tag_path.split("/") if b]
        for block in blocks:
            if "links" not in node or not isinstance(node["links"], list):
                node["links"] = []

            found = False

            for link in node["links"]:
                if self._isnode(link, block):
                    node = link
                    found = True
                    break

            if not found:
                new_child = {
                    "title": block,
                    self._IDX_LABEL: self._num_nodes,
                    self._COUNT_LABEL: 0,
                }

                node["links"].append(new_child)
                self._num_nodes += 1
                node = new_child

            node[self._COUNT_LABEL] = node.get(self._COUNT_LABEL, 0) + 1

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("Tree already finalized.")

        total = self._total_count
        if total == 0:
            self._finalized = True
            self._tree = {"links": []}
            return

        def _rec(node):
            if self._COUNT_LABEL in node:
                count = node[self._COUNT_LABEL]
                node[self._FREQ_LABEL] = count / total
                del node[self._COUNT_LABEL]
            else:
                node[self._FREQ_LABEL] = 0

            if "links" in node:
                for child in node["links"]:
                    _rec(child)

        _rec(self._tree)
        self._finalized = True


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tagged_dataset",
        type=str,
        help="HuggingFace path to the databank of tagged problems",
    )

    parser.add_argument(
        "out_tree_path", type=str, help="Path to save the output tag tree"
    )

    parser.add_argument(
        "--tags-column",
        type=str,
        default="tags",
        required=False,
        help="Databank column containing the tags",
    )

    args = parser.parse_args()

    print(f'[~] Loading databank "{args.tagged_dataset}"...')
    ds = load_dataset(args.tagged_dataset)
    print("[+] Databank loaded.")

    tag_col = args.tags_column

    # begin validate
    for name in ds.keys():
        if tag_col not in ds[name].column_names:
            print(f'[ERROR] No column "{tag_col}" in dataset "{name}".')
            return
    # end validate

    tree = Tree()

    for name in ds.keys():
        print(f'[~] Incrementing tags from "{args.tagged_dataset}:{name}"...')
        ds_local = ds[name]
        for row in tqdm(ds_local, total=ds_local.num_rows):
            tags = row[tag_col]
            if isinstance(tags, list):
                for t in tags:
                    tree.increment(t)

    tree.finalize()

    print(f"[+] Building tree finished with a total of {tree.num_nodes} nodes.")
    print("[~] Saving...")

    tree.save(args.out_tree_path)

    print("[+] All done.")


if __name__ == "__main__":
    main()
