from abc import ABC, abstractmethod
from typing import ClassVar


class ModelProcessor(ABC):
    processor: ClassVar[str | None]

    @property
    def model(self) -> str | None:
        return None

    def export_cache(self, path: str) -> None:
        raise NotImplemented

    def import_cache(self, path: str) -> None:
        raise NotImplemented

    @abstractmethod
    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float]:
        pass
