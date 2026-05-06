from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.rag.splitter.base_splitter import BaseThreeLayerSplitter


class MarkdownThreeLayerSplitter(BaseThreeLayerSplitter):
    def __init__(self, l3_size: int = 400, l3_overlap: int = 50):
        self.block_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l3_size,
            chunk_overlap=l3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    def split(self, content: list[dict[str, Any]], metadata: dict[str, Any], doc_id: str | None = None) -> list[dict[str, Any]]:
        root_doc_id = self.build_doc_id(doc_id)
        results: list[dict[str, Any]] = []
        h1_roots: dict[str, str] = {}
        h2_parents: dict[tuple[str, int], str] = {}

        for section in content:
            heading_titles = section.get("heading_titles") or ["文档内容"]
            heading_level = int(section.get("heading_level", 0))
            section_index = int(section.get("section_index", 0))
            heading_path = section.get("heading_path") or " / ".join(heading_titles)
            blocks = section.get("blocks") or []
            base_metadata = self._section_metadata(metadata, section)

            if heading_level <= 1:
                root_key = heading_titles[0]
                l1_id = h1_roots.get(root_key)
                if l1_id is None:
                    l1_id = f"{root_doc_id}_l1_{len(h1_roots)}"
                    h1_roots[root_key] = l1_id
                    results.append(
                        self.build_chunk(
                            text=self._build_l1_text(section),
                            chunk_id=l1_id,
                            parent_chunk_id="",
                            root_chunk_id=l1_id,
                            chunk_level=1,
                            metadata=base_metadata,
                            heading_path=heading_path,
                            section_type=section.get("section_type", "h1"),
                        )
                    )
                parent_l1_id = l1_id
            else:
                root_title = heading_titles[0]
                parent_l1_id = h1_roots.get(root_title)
                if parent_l1_id is None:
                    parent_l1_id = f"{root_doc_id}_l1_{len(h1_roots)}"
                    h1_roots[root_title] = parent_l1_id
                    results.append(
                        self.build_chunk(
                            text=self._build_virtual_l1_text(root_title),
                            chunk_id=parent_l1_id,
                            parent_chunk_id="",
                            root_chunk_id=parent_l1_id,
                            chunk_level=1,
                            metadata={**metadata, "heading_level": 1, "heading_titles": [root_title], "heading_path": root_title},
                            heading_path=root_title,
                            section_type="virtual_h1",
                        )
                    )

            if heading_level >= 2:
                l2_id = f"{parent_l1_id}_l2_{section_index}"
                h2_parents[(heading_path, section_index)] = l2_id
                results.append(
                    self.build_chunk(
                        text=self._build_l2_text(section),
                        chunk_id=l2_id,
                        parent_chunk_id=parent_l1_id,
                        root_chunk_id=parent_l1_id,
                        chunk_level=2,
                        metadata=base_metadata,
                        heading_path=heading_path,
                        section_type=section.get("section_type", "h2"),
                    )
                )
                self._append_l3_chunks(results, blocks, section, metadata=base_metadata, parent_l1_id=parent_l1_id, parent_l2_id=l2_id)
            else:
                self._append_l3_chunks(results, blocks, section, metadata=base_metadata, parent_l1_id=parent_l1_id, parent_l2_id=parent_l1_id)

        return results

    def _section_metadata(self, metadata: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
        return {
            **metadata,
            "heading_level": section.get("heading_level", 0),
            "heading_titles": section.get("heading_titles") or ["文档内容"],
            "heading_path": section.get("heading_path") or "文档内容",
            "section_index": section.get("section_index", 0),
            "section_type": section.get("section_type", "preface"),
        }

    def _build_l1_text(self, section: dict[str, Any]) -> str:
        titles = section.get("heading_titles") or ["文档内容"]
        title = titles[0]
        body = section.get("text", "")
        return f"# {title}\n\n{body}".strip()

    def _build_virtual_l1_text(self, title: str) -> str:
        return f"# {title}".strip()

    def _build_l2_text(self, section: dict[str, Any]) -> str:
        titles = section.get("heading_titles") or ["文档内容"]
        lines = []
        if len(titles) >= 1:
            lines.append(f"# {titles[0]}")
        if len(titles) >= 2:
            lines.append(f"## {titles[1]}")
        body = section.get("text", "")
        if body:
            lines.append(body)
        return "\n\n".join(lines).strip()

    def _append_l3_chunks(
        self,
        results: list[dict[str, Any]],
        blocks: list[dict[str, Any]],
        section: dict[str, Any],
        *,
        metadata: dict[str, Any],
        parent_l1_id: str,
        parent_l2_id: str,
    ) -> None:
        for block_index, block in enumerate(blocks):
            block_type = block.get("block_type", "paragraph")
            block_text = block.get("text", "").strip()
            if not block_text:
                continue
            block_segments = self._split_block_text(block_text)
            for segment_index, segment_text in enumerate(block_segments):
                l3_id = f"{parent_l2_id}_l3_{block_index}_{segment_index}"
                results.append(
                    self.build_chunk(
                        text=self._build_l3_text(section, segment_text),
                        chunk_id=l3_id,
                        parent_chunk_id=parent_l2_id,
                        root_chunk_id=parent_l1_id,
                        chunk_level=3,
                        metadata=metadata,
                        heading_path=section.get("heading_path", "文档内容"),
                        block_type=block_type,
                        block_index=block_index,
                        block_segment_index=segment_index,
                    )
                )

    def _split_block_text(self, text: str) -> list[str]:
        if len(text) <= self.block_splitter._chunk_size:
            return [text]
        return self.block_splitter.split_text(text)

    def _build_l3_text(self, section: dict[str, Any], block_text: str) -> str:
        titles = section.get("heading_titles") or ["文档内容"]
        lines = []
        if len(titles) >= 1:
            lines.append(f"# {titles[0]}")
        if len(titles) >= 2:
            lines.append(f"## {titles[1]}")
        lines.append(block_text)
        return "\n\n".join(lines).strip()
