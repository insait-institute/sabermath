# The dataclasses are free to import; `evaluate` and the processors are not -
# they pull in torch, vLLM and the rest of the scoring stack. Resolving them
# lazily keeps `import sabermath` cheap, so a table generator or a results
# reader does not need a GPU environment to run.
from .schemas import Branch, BranchResult, Report, Task, TaskResult

_LAZY = {
    "evaluate": ".benchmark",
    "EmbeddingProcessor": ".processors",
    "GoogleProcessor": ".processors",
    "OpenAIProcessor": ".processors",
}

__all__ = [
    "evaluate",
    "Task",
    "Report",
    "TaskResult",
    "Branch",
    "BranchResult",
    "EmbeddingProcessor",
    "GoogleProcessor",
    "OpenAIProcessor",
]


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
