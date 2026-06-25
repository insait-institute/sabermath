from abc import ABC, abstractmethod
from typing import ClassVar


class ModelProcessor(ABC):
    processor: ClassVar[str | None]

    @property
    def model(self) -> str | None:
        """Return the name of the model being evaluated."""
        return None

    def export_cache(self, path: str) -> None:
        """Export any internal cache to a file."""
        raise NotImplemented

    def import_cache(self, path: str) -> None:
        """Import any internal cache from a file."""
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
