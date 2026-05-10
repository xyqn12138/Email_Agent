from abc import ABC, abstractmethod
from pathlib import Path

from agent.utils.path_handler import get_absolute_path


class BaseLoader(ABC):
    supported_suffixes: tuple[str, ...] = ()
    document_type: str = "Document"

    def resolve_path(self, file_path: str) -> Path:
        return Path(get_absolute_path(file_path))

    def validate_path(self, file_path: str) -> Path:
        absolute_path = self.resolve_path(file_path)
        if self.supported_suffixes and absolute_path.suffix.lower() not in self.supported_suffixes:
            raise ValueError(f"Unsupported file type: {absolute_path.name.lower()}")
        return absolute_path

    def supports(self, file_path: str) -> bool:
        return not self.supported_suffixes or self.resolve_path(file_path).suffix.lower() in self.supported_suffixes

    def build_metadata(self, absolute_path: Path, **extra: object) -> dict:
        metadata = {
            "filename": absolute_path.name.lower(),
            "file_path": str(absolute_path),
        }
        metadata.update(extra)
        return metadata

    @abstractmethod
    def load(self, file_path: str):
        raise NotImplementedError
