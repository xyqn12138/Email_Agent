from abc import ABC, abstractmethod
from typing import Any
import uuid


class BaseThreeLayerSplitter(ABC):
    def build_doc_id(self, doc_id: str | None = None) -> str:
        return doc_id or str(uuid.uuid4())

    def build_chunk(self, *, text: str, chunk_id: str, parent_chunk_id: str, root_chunk_id: str, chunk_level: int, metadata: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        chunk = {
            "text": text,
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "root_chunk_id": root_chunk_id,
            "chunk_level": chunk_level,
        }
        if metadata:
            chunk.update(metadata)
        if extra:
            chunk.update(extra)
        return chunk

    def optimize_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return chunks

    @abstractmethod
    def split(self, content: Any, metadata: dict[str, Any], doc_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError
