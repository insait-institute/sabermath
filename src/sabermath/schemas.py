from dataclasses import dataclass, asdict
from typing import Literal


Task = Literal[
    "statement-statement",
    "statement-full",
    "full-full",
]


DCGVariant = Literal["exponent", "linear"]


Branch = Literal[
    "Algebra",
    "Geometry",
    "Number Theory",
    "Combinatorics",
    "Calculus and Analysis",
]


@dataclass(frozen=True)
class BranchResult:
    branch: Branch
    ndcg_at_k: float


@dataclass(frozen=True)
class TaskResult:
    task: Task
    ndcg_at_k: float
    branches: list[BranchResult]


@dataclass
class Report:
    model: str
    processor: str
    dcg_variant: DCGVariant
    k: int
    tasks: list[TaskResult]

    def to_dict(self) -> dict:
        return asdict(self)
