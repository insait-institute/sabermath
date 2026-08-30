"""Read the math-vs-word figure's model registries without running it.

The latency figure must name, mark and colour a model exactly as
``experiments/math-vs-word/plot_hist.py`` does, or the two figures cannot be
read side by side. Copying the entries across would let them drift, and
importing plot_hist.py is not an option - it is a script, not a module: at
import time it parses argv, loads a HuggingFace dataset and reads
``similarities/``.

Its three registries are pure dict literals, so lift them out with ``ast``
instead. That is a read of the same source of truth, and a rename there shows
up here on the next run.
"""

import ast
from pathlib import Path

PLOT_HIST = Path(__file__).resolve().parents[2] / "experiments/math-vs-word/plot_hist.py"


def _literal_dict(name: str) -> dict:
    for node in ast.parse(PLOT_HIST.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return {
                key.value: ast.literal_eval(value)
                for key, value in zip(node.value.keys, node.value.values)
                # Skips the MATH_TOKEN_RATIO_MODEL_ID entry, whose key is a
                # Name rather than a literal - it is a reference line, not a
                # model, and this figure has no equivalent.
                if isinstance(key, ast.Constant)
            }
    raise SystemExit(f"{name} is not a module-level dict in {PLOT_HIST.name}")


def display_names() -> dict:
    return _literal_dict("DEFAULT_MODEL_DISPLAY_NAMES")


def marker_symbols() -> dict:
    return _literal_dict("DEFAULT_MODEL_MARKER_SYMBOLS")


def colors() -> dict:
    return _literal_dict("DEFAULT_MODEL_COLORS")
