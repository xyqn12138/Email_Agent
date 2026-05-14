from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.rag.splitter.base_splitter import BaseThreeLayerSplitter

LEVEL_CHAPTER = 1
LEVEL_SECTION = 2


class MarkdownThreeLayerSplitter(BaseThreeLayerSplitter):
    def __init__(self, l3_size: int = 1500, l3_overlap: int = 150, l4_size: int = 400, l4_overlap: int = 60):
        self.l3_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l3_size,
            chunk_overlap=l3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        self.l4_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l4_size,
            chunk_overlap=l4_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    def split(
        self,
        content: list[dict[str, Any]],
        metadata: dict[str, Any],
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        root_doc_id = self.build_doc_id(doc_id)
        doc_prefix = root_doc_id[:8]
        results: list[dict[str, Any]] = []
        l1_nodes: dict[str, str] = {}
        l2_nodes: dict[str, str] = {}
        seq = {1: 0, 2: 0, 3: 0, 4: 0}

        for section in content:
            logical_level = int(section.get("logical_level", LEVEL_CHAPTER))
            node_id = section.get("node_id") or ""
            blocks = section.get("blocks") or []
            base_meta = self._base_metadata(metadata, section)
            base_meta["doc_id"] = root_doc_id

            if logical_level <= LEVEL_CHAPTER:
                l1_id = l1_nodes.get(node_id)
                if l1_id is None:
                    l1_id = f"{doc_prefix}_L1_{seq[1]:04d}"
                    seq[1] += 1
                    l1_nodes[node_id] = l1_id
                    results.append(self._make_l1(l1_id, section, base_meta))
            else:
                l1_id = self._find_or_create_l1(
                    results, l1_nodes, section, metadata, doc_prefix, seq,
                )

            if logical_level >= LEVEL_SECTION:
                l2_id = l2_nodes.get(node_id)
                if l2_id is None:
                    l2_id = f"{doc_prefix}_L2_{seq[2]:04d}"
                    seq[2] += 1
                    l2_nodes[node_id] = l2_id
                    results.append(self._make_l2(l2_id, l1_id, section, base_meta))
                self._append_l3_l4(results, blocks, section, base_meta, l1_id, l2_id, doc_prefix, seq)
            else:
                self._append_l3_l4(results, blocks, section, base_meta, l1_id, l1_id, doc_prefix, seq)

        return results

    def _base_metadata(self, metadata: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
        return {
            **metadata,
            "title": section.get("title") or "",
            "title_path": section.get("title_path") or "",
            "content_type": section.get("content_type", "chapter"),
            "logical_level": int(section.get("logical_level", LEVEL_CHAPTER)),
            "source_start_line": int(section.get("source_start_line") or 0),
            "source_end_line": int(section.get("source_end_line") or 0),
            "page_number": int(section.get("page_start") or 0),
        }

    def _make_l1(self, l1_id: str, section: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        title = section.get("title") or "文档内容"
        return self.build_chunk(
            text=title,
            chunk_id=l1_id,
            parent_chunk_id="",
            root_chunk_id=l1_id,
            chunk_level=1,
            metadata=meta,
        )

    def _find_or_create_l1(
        self,
        results: list[dict[str, Any]],
        l1_nodes: dict[str, str],
        section: dict[str, Any],
        metadata: dict[str, Any],
        doc_prefix: str,
        seq: dict[int, int],
    ) -> str:
        parent_node_id = section.get("parent_node_id") or ""
        # Direct match
        if parent_node_id and parent_node_id in l1_nodes:
            return l1_nodes[parent_node_id]
        # Slug match: L1 node_id = "node_{idx}_{slug}", parent = "node_{slug}"
        if parent_node_id:
            parent_slug = parent_node_id.replace("node_", "", 1)
            # Strip leading index if present (e.g., "3_第1章" → "第1章")
            for nid, lid in l1_nodes.items():
                if nid.startswith("node_"):
                    nid_slug = nid.split("_", 2)[-1] if nid.count("_") >= 2 else nid[5:]
                    if nid_slug == parent_slug:
                        return lid
        titles = (section.get("title_path") or section.get("title") or "文档内容").split(" / ")
        root_title = titles[0]
        for nid, lid in l1_nodes.items():
            if root_title in nid:
                return lid
        l1_id = f"{doc_prefix}_L1_{seq[1]:04d}"
        seq[1] += 1
        fake_section = {
            "title": root_title,
            "title_path": root_title,
            "node_id": f"virtual_{root_title}",
            "parent_node_id": "",
            "content_type": "chapter",
            "logical_level": LEVEL_CHAPTER,
            "body": "",
            "text": "",
        }
        l1_nodes[fake_section["node_id"]] = l1_id
        meta = {**metadata, "title": root_title, "title_path": root_title, "logical_level": LEVEL_CHAPTER}
        results.append(self._make_l1(l1_id, fake_section, meta))
        return l1_id

    def _make_l2(self, l2_id: str, l1_id: str, section: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        title = section.get("title") or "文档内容"
        return self.build_chunk(
            text=title,
            chunk_id=l2_id,
            parent_chunk_id=l1_id,
            root_chunk_id=l1_id,
            chunk_level=2,
            metadata=meta,
        )

    def _append_l3_l4(
        self,
        results: list[dict[str, Any]],
        blocks: list[dict[str, Any]],
        section: dict[str, Any],
        meta: dict[str, Any],
        l1_id: str,
        l2_id: str,
        doc_prefix: str,
        seq: dict[int, int],
    ) -> None:
        groups = self._group_paragraphs(blocks)
        for group in groups:
            group_text = group["text"].strip()
            if not group_text:
                continue
            group_meta = self._block_metadata(meta, group)
            group_meta["merged_from"] = group.get("merged_from", [])

            if group["block_type"] in ("table", "code", "math", "image"):
                l3_id = f"{doc_prefix}_L3_{seq[3]:04d}"
                seq[3] += 1
                results.append(self._make_l3(l3_id, l2_id, l1_id, section, group_text, group_meta))
                continue

            l3_segments = self._split_text(group_text, self.l3_splitter)
            for seg_text in l3_segments:
                l3_id = f"{doc_prefix}_L3_{seq[3]:04d}"
                seq[3] += 1
                results.append(self._make_l3(l3_id, l2_id, l1_id, section, seg_text, group_meta))
                l4_segments = self._split_text(seg_text, self.l4_splitter)
                if len(l4_segments) > 1:
                    for l4_text in l4_segments:
                        l4_id = f"{doc_prefix}_L4_{seq[4]:04d}"
                        seq[4] += 1
                        results.append(self._make_l4(l4_id, l3_id, l1_id, section, l4_text, group_meta))

    def _group_paragraphs(self, blocks: list[dict[str, Any]], min_size: int = 1500) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        buf_texts: list[str] = []
        buf_merged: list[list[int]] = []
        buf_start = 0
        buf_end = 0
        carry_context = ""

        def flush() -> None:
            nonlocal carry_context
            if not buf_texts:
                return
            merged_text = "\n\n".join(buf_texts)
            if carry_context:
                merged_text = carry_context + "\n\n" + merged_text
                carry_context = ""
            groups.append({
                "text": merged_text,
                "block_type": "paragraph",
                "source_start_line": buf_start,
                "source_end_line": buf_end,
                "merged_from": buf_merged,
            })
            plain = "\n\n".join(buf_texts)
            if len(plain) > 120:
                carry_context = plain[-120:]
            buf_texts.clear()
            buf_merged.clear()

        for block in blocks:
            bt = block.get("block_type", "paragraph")
            if bt in ("table", "code", "math"):
                flush()
                groups.append(block)
                continue

            block_text = block.get("text", "").strip()
            if not block_text:
                continue

            if bt == "image":
                if buf_texts:
                    flush()
                if groups:
                    prev = groups[-1]
                    prev["text"] += "\n\n" + block_text
                    prev["source_end_line"] = int(block.get("source_end_line") or prev["source_end_line"])
                    if "merged_from" not in prev:
                        prev["merged_from"] = []
                    prev["merged_from"].append(block.get("merged_from", [[0, 0]]))
                    continue

            if not buf_texts:
                buf_start = int(block.get("source_start_line") or 0)
            buf_end = int(block.get("source_end_line") or 0)
            buf_texts.append(block_text)
            buf_merged.append(block.get("merged_from", [[0, 0]]))

            if sum(len(t) for t in buf_texts) >= min_size:
                flush()

        flush()
        return groups

    def _make_l3(
        self,
        l3_id: str,
        l2_id: str,
        l1_id: str,
        section: dict[str, Any],
        text: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return self.build_chunk(
            text=self._prepend_heading_prefix(section, text, max_levels=3),
            chunk_id=l3_id,
            parent_chunk_id=l2_id,
            root_chunk_id=l1_id,
            chunk_level=3,
            metadata=meta,
        )

    def _make_l4(
        self,
        l4_id: str,
        l3_id: str,
        l1_id: str,
        section: dict[str, Any],
        text: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return self.build_chunk(
            text=self._prepend_heading_prefix(section, text, max_levels=4),
            chunk_id=l4_id,
            parent_chunk_id=l3_id,
            root_chunk_id=l1_id,
            chunk_level=4,
            metadata=meta,
        )

    def _block_metadata(self, meta: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
        return {
            **meta,
            "block_type": block.get("block_type", "paragraph"),
            "source_start_line": int(block.get("source_start_line") or meta.get("source_start_line") or 0),
            "source_end_line": int(block.get("source_end_line") or meta.get("source_end_line") or 0),
            "merged_from": block.get("merged_from") or [],
            "title": block.get("title") or meta.get("title") or "",
            "page_number": int(block.get("page_start") or meta.get("page_number") or 0),
        }

    def _prepend_heading_prefix(self, section: dict[str, Any], text: str, max_levels: int) -> str:
        titles = (section.get("title_path") or section.get("title") or "").split(" / ")
        lines = []
        for i, title in enumerate(titles[:max_levels], start=1):
            lines.append(f"{'#' * i} {title}")
        lines.append(text)
        return "\n\n".join(lines).strip()

    def _split_text(self, text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
        if len(text) <= splitter._chunk_size:
            return [text]
        return splitter.split_text(text)
    
if __name__ == "__main__":
    from agent.rag.Loader.md_loader import MarkdownLoader
    loader = MarkdownLoader()
    docs = loader.load(r"data\算法基础\算法基础.md")
    splitter = MarkdownThreeLayerSplitter()
    meta = {"filename": "算法基础", "file_path": r"data\算法基础\算法基础.md", "doc_id": "test-doc-id"}
    chunks = splitter.split(docs, meta)
    print(f"总分块数: {len(chunks)}")

    for chunk in chunks:
        if chunk["chunk_level"] <= 2:
            print(chunk)

        

