
from __future__ import annotations

import ast
from pathlib import Path

from .aggregate import NON_EMBEDDING_METHODS

LOAD_MODELS_PY = Path(__file__).with_name("load_models.py")
REGISTRY_PY = Path(__file__).resolve().parents[1] / "registry.py"


def literal_env(path: Path, env: dict | None = None) -> dict:
    env = {} if env is None else env
    tree = ast.parse(Path(path).read_text())

    def evaluate(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Subscript):
            return evaluate(node.value)[ast.literal_eval(node.slice)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {
                ast.literal_eval(k): evaluate(v)
                for k, v in zip(node.keys, node.values)
            }
        return ast.literal_eval(node)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                env[target.id] = evaluate(node.value)
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
    return env


def allowed_models(
    load_models_py: Path = LOAD_MODELS_PY,
    extra_sources: list[Path] | None = None,
) -> list[str]:
    env: dict[str, object] = {}
    sources = [REGISTRY_PY] if extra_sources is None else list(extra_sources)
    for source in sources:
        if Path(source).exists():
            literal_env(Path(source), env)
    literal_env(Path(load_models_py), env)

    models = env.get("ALLOWED_MODELS")
    if not models:
        raise SystemExit(f"Could not read ALLOWED_MODELS from {load_models_py}")
    return list(models)


def all_methods(
    load_models_py: Path = LOAD_MODELS_PY,
    extra_sources: list[Path] | None = None,
) -> list[str]:
    return allowed_models(load_models_py, extra_sources) + NON_EMBEDDING_METHODS
