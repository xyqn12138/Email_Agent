import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.models.embedding_model import EmbeddingModel
from agent.rag.Loader.base_loader import BaseLoader
from agent.rag.splitter.base_splitter import BaseThreeLayerSplitter
from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_absolute_path

logger = get_logger()

_MD_IMAGE_RE = re.compile(r"!\[.*?\]\(([^)]+)\)")


def extract_image_paths(text: str) -> list[str]:
    return _MD_IMAGE_RE.findall(text)


def strip_images(text: str) -> str:
    return _MD_IMAGE_RE.sub("", text).strip()


@dataclass(frozen=True)
class PreparedDocument:
    pipeline_name: str
    loaded_content: Any
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any]
    content_fingerprint: str


class BaseDocumentPipeline:
    def __init__(self, *, name: str, loader: BaseLoader, splitter: BaseThreeLayerSplitter):
        self.name = name
        self.loader = loader
        self.splitter = splitter

    def supports(self, file_path: str) -> bool:
        return self.loader.supports(file_path)

    def prepare(self, file_path: str) -> PreparedDocument:
        absolute_path = self.loader.validate_path(file_path)
        loaded_content = self.loader.load(file_path)
        metadata = self._build_document_metadata(absolute_path, loaded_content)
        content_fingerprint = self._build_content_fingerprint(loaded_content)
        doc_id = self._build_doc_id(content_fingerprint, metadata)
        split_content = self.get_split_content(loaded_content)
        chunks = self.splitter.split(split_content, metadata, doc_id=doc_id)
        optimized_chunks = self.splitter.optimize_chunks(chunks)
        return PreparedDocument(
            pipeline_name=self.name,
            loaded_content=loaded_content,
            chunks=optimized_chunks,
            metadata=metadata,
            content_fingerprint=content_fingerprint,
        )

    def _build_document_metadata(self, absolute_path: Path, loaded_content: Any) -> dict[str, Any]:
        metadata = {
            "filename": absolute_path.name.lower(),
            "file_path": str(absolute_path),
        }
        if isinstance(loaded_content, list) and loaded_content:
            first_item = loaded_content[0]
            for field in ("filename", "file_path"):
                value = first_item.get(field)
                if value is not None:
                    metadata[field] = value
        return metadata

    def _build_content_fingerprint(self, loaded_content: Any) -> str:
        content_text = self.serialize_loaded_content(loaded_content)
        return hashlib.md5(content_text.encode("utf-8")).hexdigest()

    def _build_doc_id(self, content_fingerprint: str, metadata: dict[str, Any]) -> str:
        filename = metadata.get("filename") or "unknown"
        return f"{filename}_{content_fingerprint[:12]}"

    def serialize_loaded_content(self, loaded_content: Any) -> str:
        if isinstance(loaded_content, list):
            return "\n".join(item.get("text", "") for item in loaded_content)
        return str(loaded_content or "")

    def get_split_content(self, loaded_content: Any) -> Any:
        raise NotImplementedError


class MarkdownDocumentPipeline(BaseDocumentPipeline):
    def get_split_content(self, loaded_content: Any) -> list[dict[str, Any]]:
        return loaded_content


def _build_markdown_pipeline() -> MarkdownDocumentPipeline:
    from agent.rag.Loader.md_loader import MarkdownLoader
    from agent.rag.splitter.md_splitter import MarkdownThreeLayerSplitter

    return MarkdownDocumentPipeline(
        name="markdown",
        loader=MarkdownLoader(),
        splitter=MarkdownThreeLayerSplitter(),
    )


class DataEmbedding:
    def __init__(self, model_name: str = "local", dimensions: int = 1024):
        from agent.rag.milvus_manage import MilvusManage

        self.embedding_model = EmbeddingModel(model_name=model_name, dimensions=dimensions)
        self.milvus_manager = MilvusManage()
        self.md5_file = get_absolute_path("src/agent/data/processed_md5.txt")
        self.insert_batch_size = max(1, int(os.getenv("RAG_INSERT_BATCH_SIZE", "32")))
        os.makedirs(os.path.dirname(self.md5_file), exist_ok=True)

    def _is_processed(self, md5_val: str) -> bool:
        if not os.path.exists(self.md5_file):
            return False
        with open(self.md5_file, "r", encoding="utf-8") as f:
            processed_md5s = f.read().splitlines()
        return md5_val in processed_md5s

    def _mark_as_processed(self, md5_val: str):
        with open(self.md5_file, "a", encoding="utf-8") as f:
            f.write(md5_val + "\n")

    def _chunk_items(self, items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _ensure_collection(self, dense_dim: int):
        if not self.milvus_manager.has_collection():
            self.milvus_manager.create_collection(dense_dim=dense_dim)

    def _filter_uninserted_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_ids = [chunk["chunk_id"] for chunk in chunks]
        existing: set[str] = set()
        for i in range(0, len(all_ids), 100):
            batch = all_ids[i : i + 100]
            rows = self.milvus_manager.query_by_chunk_ids(batch, output_fields=["chunk_id"])
            for row in rows:
                existing.add(row["chunk_id"])
        return [chunk for chunk in chunks if chunk["chunk_id"] not in existing]

    def _build_milvus_batch(self, chunks: list[dict[str, Any]], dense_embeddings: list[list[float]], start_idx: int) -> list[dict]:
        milvus_data = []
        for offset, chunk in enumerate(chunks):
            original_text = chunk["text"]
            image_paths = extract_image_paths(original_text)
            clean_text = strip_images(original_text)
            chunk_data = {
                "text": clean_text,
                "text_dense": dense_embeddings[offset],
                "doc_id": chunk.get("doc_id") or "",
                "filename": chunk.get("filename") or "",
                "file_path": chunk.get("file_path") or "",
                "chunk_id": chunk["chunk_id"],
                "parent_chunk_id": chunk.get("parent_chunk_id") or "",
                "root_chunk_id": chunk.get("root_chunk_id") or "",
                "chunk_level": int(chunk["chunk_level"]),
                "title_path": chunk.get("title_path") or "",
                "title": chunk.get("title") or "",
                "content_type": chunk.get("content_type") or "",
                "page_number": int(chunk.get("page_number") or 0),
            }
            if image_paths:
                chunk_data["image_paths"] = ";".join(image_paths)
            milvus_data.append(chunk_data)
        return milvus_data

    def process_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        content_fingerprint: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        if self._is_processed(content_fingerprint):
            logger.info(
                f"Document '{metadata.get('filename', 'unknown')}' with MD5 {content_fingerprint} already processed. Skipping."
            )
            return

        if not chunks:
            logger.warning("No chunks prepared for ingestion, skipping.")
            return

        pending_chunks = self._filter_uninserted_chunks(chunks)

        if not pending_chunks:
            self._mark_as_processed(content_fingerprint)
            logger.info(f"Document '{metadata.get('filename', 'unknown')}' already fully inserted. Marked as processed.")
            return

        logger.info(f"Preparing to insert {len(pending_chunks)} pending chunks out of {len(chunks)} total chunks.")

        use_async = self.embedding_model.model_name in ("dashscope", "zhipuai")

        for batch_index, batch_chunks in enumerate(self._chunk_items(pending_chunks, self.insert_batch_size), start=1):
            logger.info(f"Embedding batch {batch_index}, chunk count: {len(batch_chunks)}")
            texts = [strip_images(chunk["text"]) for chunk in batch_chunks]
            if use_async:
                dense_embeddings = asyncio.run(self.embedding_model.async_embed_documents(texts))
            else:
                dense_embeddings = self.embedding_model.embed_documents(texts)
            self._ensure_collection(dense_dim=len(dense_embeddings[0]))
            milvus_data = self._build_milvus_batch(batch_chunks, dense_embeddings, start_idx=(batch_index - 1) * self.insert_batch_size)
            logger.info(f"Inserting batch {batch_index} into Milvus...")
            self.milvus_manager.insert(milvus_data)

        self._mark_as_processed(content_fingerprint)
        logger.info("Insertion complete and MD5 marked.")

    def close(self):
        self.milvus_manager.disconnect()


_PDF_SUFFIXES = {".pdf"}


class RAGPipelineService:
    def __init__(
        self,
        model_name: str = "local",
        dimensions: int = 1024,
        data_embedding: DataEmbedding | None = None,
    ):
        self._md_pipeline = _build_markdown_pipeline()
        self.data_embedding = data_embedding or DataEmbedding(model_name=model_name, dimensions=dimensions)

    def ingest_file(self, file_path: str) -> PreparedDocument:
        suffix = Path(file_path).suffix.lower()
        if suffix in _PDF_SUFFIXES:
            from agent.rag.Loader.minerU import MinerUParser

            parser = MinerUParser()
            output_dir = parser.parse(file_path)
            md_candidates = list(output_dir.rglob("*.md"))
            if not md_candidates:
                raise FileNotFoundError(f"MinerU did not produce a markdown file in {output_dir}")
            file_path = str(md_candidates[0])
            logger.info(f"MinerU converted PDF to: {file_path}")

        prepared_document = self._md_pipeline.prepare(file_path)
        logger.info(f"Using pipeline '{self._md_pipeline.name}' for file '{file_path}'.")
        self.data_embedding.process_chunks(
            prepared_document.chunks,
            content_fingerprint=prepared_document.content_fingerprint,
            metadata=prepared_document.metadata,
        )
        return prepared_document

    def close(self) -> None:
        self.data_embedding.close()


if __name__ == "__main__":
    service = RAGPipelineService(model_name="dashscope")
    try:
        prepared = service.ingest_file(r"data\计算机操作系统  第4版·微课视频\计算机操作系统  第4版·微课视频.md")
        print(prepared.pipeline_name)
        print(prepared.metadata)
        print(f"chunks: {len(prepared.chunks)}")
        print(prepared.chunks[0])
    finally:
        service.close()
