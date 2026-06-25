import json
import numpy as np


class Tree:
    def __init__(self, tree: dict):
        self._tree = tree
        # index -> node
        self._idxs = {}
        self._max_idx = -1
        # path -> node
        self._path_index = {}

        def _walk_rec(node, path: str = ""):
            if "title" not in node:
                raise ValueError(f'Missing "title" field.')

            title = node["title"]
            new_path = f"{path}/{title}"
            self._path_index[new_path] = node

            if "idx" not in node:
                raise ValueError(f'No "idx" in node "{title}"')

            idx = node["idx"]
            if not isinstance(idx, int):
                raise ValueError(f"Invalid index {idx}.")

            if idx in self._idxs:
                raise ValueError(f"Duplication of index {idx}.")

            self._idxs[idx] = node
            if idx > self._max_idx:
                self._max_idx = idx

            for link in node.get("links", []):
                _walk_rec(link, new_path)

        _walk_rec(self._tree)
        self._len = self._max_idx + 1

        # begin validate
        for i in range(self._len):
            if i not in self._idxs:
                raise ValueError(f"Missing index {i}.")
        # end validate

    @property
    def num_nodes(self) -> int:
        return self._len

    def get_all_valid_paths(self) -> list[str]:
        return list(self._path_index.keys())

    def split_node_path(self, node_path: str) -> list[str]:
        return [b for b in node_path.split("/") if b]

    def is_valid_path(self, path: str) -> bool:
        return f"/{path.strip("/")}" in self._path_index

    def node_by_index(self, idx: int) -> dict:
        if idx < 0:
            raise ValueError("Index must be non-negative.")
        if idx >= self.num_nodes:
            raise ValueError("Index out of range.")

        return self._idxs[idx]

    def node_by_path(self, node_path: str) -> dict | None:
        return self._path_index.get(f"/{node_path.strip('/')}", None)

    def node_index_by_path(self, node_path: str) -> int:
        node = self.node_by_path(node_path)
        return node.get("idx", -1) if node else -1

    @classmethod
    def load_tree(cls, path: str):
        with open(path, "r") as file:
            json_tree = json.load(file)
            return cls(json_tree)


load_tree = Tree.load_tree
