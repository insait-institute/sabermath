import json


class TagTree:
    def __init__(self, tree: dict):
        if ("title" not in tree) or (not tree["title"]):
            raise ValueError(
                "Invalid tree - root directory must have non-empty 'title'"
            )

        self._root_node = tree

    @property
    def root(self) -> str:
        return self._root_node["title"]

    def ls(self, path: str | None = None) -> list[str]:
        if path is None:
            return self._ls_node(self._root_node)

        if path == "/":
            return [self.root]

        blocks = path.split("/")

        if not path.startswith(f"/{self.root}"):
            raise ValueError(f"Path must begin with '/{self.root}'")

        blocks = blocks[2:]
        node = self._root_node

        for block in blocks:
            if ("links" not in node) or (not isinstance(node["links"], list)):
                raise RuntimeError(
                    f"Misformated node '{node["title"]} lacks 'links' item"
                )

            for link in node["links"]:
                if "title" in link and link["title"] == block:
                    node = link
                    break

        return self._ls_node(node)

    def _ls_node(self, node) -> list[str]:
        if ("links" not in node) or (not isinstance(node["links"], list)):
            return []

        subnodes = [subnode["title"] for subnode in node["links"] if "title" in subnode]

        return subnodes

    @classmethod
    def load(cls, path: str) -> "TagTree":
        with open(path, "r") as file:
            data = json.load(file)
            if isinstance(data, list):
                raise RuntimeError("Data must be an object, not an array.")
            return cls(data)


load_tag_tree = TagTree.load
