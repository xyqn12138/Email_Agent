import json
import re
from pathlib import Path
from typing import Any

from agent.rag.Loader.base_loader import BaseLoader

LEVEL_PART = 0
LEVEL_CHAPTER = 1
LEVEL_SECTION = 2
LEVEL_SUBSECTION = 3
LEVEL_SUBSUB = 4

_RE_CHAPTER = re.compile(r"^第[0-9一二三四五六七八九十百千]+[章节篇]\s*")
_RE_NUM_DOTTED = re.compile(r"^(\d+(?:\.\d+)*)\s+")
_RE_NUM_PERIOD = re.compile(r"^(\d+)\.[\s]*")
_RE_NUM_PAREN = re.compile(r"^[（(]?\d+[）)]\s*")
_RE_CHINESE_NUM = re.compile(r"^[一二三四五六七八九十]+[、．.]\s*")
_RE_TOC_ITEM = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)\s+\d+\s*$")
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownLoader(BaseLoader):
    document_type = "Markdown"
    supported_suffixes = (".md", ".markdown")

    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max(1, max_chunk_size)

    def load(self, file_path: str, line_to_page_map: list[int] | None = None):
        absolute_path = self.validate_path(file_path)
        text = absolute_path.read_text(encoding="utf-8")
        if line_to_page_map is None:
            content_list_path = self._find_content_list(absolute_path)
            if content_list_path:
                line_to_page_map = self.build_line_page_map(text, content_list_path)
        toc_entries, toc_end = self._extract_toc_entries(text)
        sections = self._parse_sections(text, toc_entries, toc_end)
        outline = self._build_outline(sections)
        results = []
        for section_index, section in enumerate(sections):
            if line_to_page_map:
                page_range = self._line_range_to_page_range(
                    section["source_start_line"],
                    section["source_end_line"],
                    line_to_page_map,
                )
                section["page_start"] = page_range[0]
                section["page_end"] = page_range[1]
                for block in section["blocks"]:
                    block_page = self._line_range_to_page_range(
                        block["source_start_line"],
                        block["source_end_line"],
                        line_to_page_map,
                    )
                    block["page_start"] = block_page[0]
                    block["page_end"] = block_page[1]
            else:
                section["page_start"] = 0
                section["page_end"] = 0
                for block in section["blocks"]:
                    block["page_start"] = 0
                    block["page_end"] = 0
            results.append(
                self.build_metadata(
                    absolute_path,
                    text=section["body"],
                    logical_level=section["logical_level"],
                    title_path=" / ".join(section["path"]),
                    title_titles=section["path"],
                    node_id=section["node_id"],
                    parent_node_id=section["parent_node_id"],
                    content_type=section["content_type"],
                    title=section["title"],
                    source_start_line=section["source_start_line"],
                    source_end_line=section["source_end_line"],
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    outline=outline,
                    blocks=section["blocks"],
                    section_index=section_index,
                )
            )
        return results

    def _classify_heading(self, title: str) -> int:
        if _RE_CHAPTER.match(title):
            return LEVEL_CHAPTER
        num_match = _RE_NUM_DOTTED.match(title)
        if num_match:
            depth = num_match.group(1).count(".")
            return min(LEVEL_SECTION + depth - 1, LEVEL_SUBSUB)
        if _RE_NUM_PERIOD.match(title):
            return LEVEL_SUBSECTION
        if _RE_NUM_PAREN.match(title):
            return LEVEL_SUBSUB
        if _RE_CHINESE_NUM.match(title):
            return LEVEL_SUBSUB
        return LEVEL_CHAPTER

    @staticmethod
    def _find_content_list(md_path: Path) -> Path | None:
        parent = md_path.parent
        for pattern in ("content_list.json", "*_content_list.json"):
            matches = sorted(parent.glob(pattern))
            if matches:
                return matches[0]
        return None

    @staticmethod
    def build_line_page_map(text: str, content_list_path: Path) -> list[int]:
        with open(content_list_path, encoding="utf-8") as f:
            content_list = json.load(f)

        lines = text.splitlines()
        total_lines = len(lines)
        lpm = [0] * (total_lines + 1)

        def norm(t: str) -> str:
            return re.sub(r"\s+", "", t).lower()

        cursor = 1
        for item in content_list:
            raw_text = item.get("text", "").strip()
            if not raw_text:
                continue
            page_idx = item.get("page_idx", 0)
            item_norm = norm(raw_text)
            best_score = 0.0
            best_line = -1
            search_start = cursor
            search_end = min(total_lines, search_start + 200)
            for ln in range(search_start, search_end + 1):
                chunk = "\n".join(lines[cursor:ln + 1])
                chunk_norm = norm(chunk)
                if not chunk_norm:
                    continue
                overlap = 0
                pos = 0
                while True:
                    idx = item_norm.find(chunk_norm[pos:pos + 50], pos)
                    if idx < 0:
                        break
                    end = min(len(item_norm), idx + 50, len(chunk_norm) - pos)
                    overlap += end - pos
                    pos = end
                    if pos >= len(chunk_norm):
                        break
                score = overlap / max(len(item_norm), 1)
                if score > best_score:
                    best_score = score
                    best_line = ln
                if best_score > 0.6:
                    break
            if best_line > 0 and best_score > 0.2:
                for ln in range(cursor, best_line + 1):
                    if lpm[ln] == 0:
                        lpm[ln] = page_idx
                cursor = best_line + 1
            else:
                if cursor <= total_lines and lpm[cursor] == 0:
                    lpm[cursor] = page_idx
                cursor += 1

        last = 0
        for i in range(1, total_lines + 1):
            if lpm[i] > 0:
                last = lpm[i]
            elif last > 0:
                lpm[i] = last

        return lpm

    @staticmethod
    def _line_range_to_page_range(start_line: int, end_line: int, lpm: list[int]) -> tuple[int, int]:
        if not lpm or start_line <= 0:
            return (0, 0)
        start_line = max(1, min(start_line, len(lpm) - 1))
        end_line = max(start_line, min(end_line, len(lpm) - 1))
        page_start = lpm[start_line]
        page_end = lpm[end_line]
        if page_start == 0 and page_end > 0:
            page_start = page_end
        if page_end == 0 and page_start > 0:
            page_end = page_start
        return (page_start, page_end)

    def _extract_toc_entries(self, text: str) -> list[dict[str, Any]]:
        lines = text.splitlines()
        toc_start = None
        for i, line in enumerate(lines):
            if line.strip() == "# 目录":
                toc_start = i
                break
        if toc_start is None:
            return []

        entries: list[dict[str, Any]] = []
        toc_end = len(lines)
        seen_norms: set[str] = set()

        def _norm(title: str) -> str:
            return re.sub(r"\s+\d+\s*$", "", title).strip()

        for i in range(toc_start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            heading_match = _RE_HEADING.match(stripped)
            if heading_match:
                title = heading_match.group(2).strip()
                logical_level = self._classify_heading(title)
                norm = _norm(title)
                if norm in seen_norms:
                    toc_end = i
                    break
                seen_norms.add(norm)
                entries.append({
                    "title": title,
                    "logical_level": logical_level,
                    "line_no": i + 1,
                })
                continue
            toc_match = _RE_TOC_ITEM.match(stripped)
            if toc_match:
                num_str = toc_match.group(1)
                title_text = toc_match.group(2).strip()
                depth = num_str.count(".")
                level = min(LEVEL_SECTION + depth - 1, LEVEL_SUBSUB)
                full_title = f"{num_str} {title_text}"
                entries.append({
                    "title": full_title,
                    "logical_level": level,
                    "line_no": i + 1,
                })
                continue
            cleaned = re.sub(r"\s+\d+\s*$", "", stripped)
            cleaned = re.sub(r"^#+\s*", "", cleaned)
            if cleaned and _RE_CHAPTER.match(cleaned):
                norm = _norm(cleaned)
                if norm in seen_norms:
                    toc_end = i
                    break
                seen_norms.add(norm)
                entries.append({
                    "title": cleaned,
                    "logical_level": LEVEL_CHAPTER,
                    "line_no": i + 1,
                })

        return entries, toc_end

    def _parse_sections(self, text: str, toc_entries: list[dict[str, Any]], toc_end: int = 0) -> list[dict[str, Any]]:
        lines = text.splitlines()
        sections: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        preface_lines: list[tuple[int, str]] = []
        fence_marker: str | None = None
        toc_end_line = toc_end

        if toc_entries:
            for entry in toc_entries:
                if entry["logical_level"] == LEVEL_CHAPTER:
                    toc_end_line = max(toc_end_line, entry["line_no"])

        def build_section(node: dict[str, Any]) -> None:
            body = self._clean_body([line for _, line in node["lines"]])
            if not body:
                return
            path = [item["title"] for item in node["path_nodes"]]
            node_id = self._build_node_id(len(sections), path)
            parent_node_id = self._build_node_id(-1, path[:-1]) if len(path) > 1 else ""
            source_lines = [line_no for line_no, _ in node["lines"]]
            logical_level = node["logical_level"]
            content_type = self._level_to_content_type(logical_level)
            sections.append({
                "logical_level": logical_level,
                "path": path,
                "body": body,
                "blocks": self._split_blocks(node["lines"], path, node_id),
                "content_type": content_type,
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "title": node["title"],
                "source_start_line": min(source_lines) if source_lines else node["line_no"],
                "source_end_line": max(source_lines) if source_lines else node["line_no"],
            })

        def build_preface() -> None:
            nonlocal preface_lines
            body = self._clean_body([line for _, line in preface_lines])
            if not body:
                preface_lines = []
                return
            path = ["文档内容"]
            node_id = self._build_node_id(len(sections), path)
            source_lines = [line_no for line_no, _ in preface_lines]
            sections.append({
                "logical_level": LEVEL_CHAPTER,
                "path": path,
                "body": body,
                "blocks": self._split_blocks(preface_lines, path, node_id),
                "content_type": "preface",
                "node_id": node_id,
                "parent_node_id": "",
                "title": "文档内容",
                "source_start_line": min(source_lines),
                "source_end_line": max(source_lines),
            })
            preface_lines = []

        for line_no, line in enumerate(lines, start=1):
            if toc_end_line > 0 and line_no <= toc_end_line:
                continue

            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if fence_marker is None:
                    fence_marker = marker
                elif marker == fence_marker:
                    fence_marker = None

            if fence_marker is not None:
                if stack:
                    stack[-1]["lines"].append((line_no, line))
                else:
                    preface_lines.append((line_no, line))
                continue

            heading_match = _RE_HEADING.match(stripped)
            if heading_match:
                raw_level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                logical_level = self._classify_heading(title)

                build_preface()
                while stack and stack[-1]["logical_level"] >= logical_level:
                    build_section(stack.pop())
                path_nodes = stack + [{"title": title, "logical_level": logical_level}]
                stack.append({
                    "title": title,
                    "logical_level": logical_level,
                    "raw_level": raw_level,
                    "line_no": line_no,
                    "lines": [],
                    "path_nodes": path_nodes,
                })
                continue

            if stack:
                stack[-1]["lines"].append((line_no, line))
            else:
                preface_lines.append((line_no, line))

        build_preface()
        while stack:
            build_section(stack.pop())

        sections.sort(key=lambda s: (s["source_start_line"], s["source_end_line"], s["logical_level"]))

        if not sections and text.strip():
            body_lines = [(line_no, line) for line_no, line in enumerate(lines, start=1)]
            path = ["文档内容"]
            node_id = self._build_node_id(0, path)
            sections.append({
                "logical_level": LEVEL_CHAPTER,
                "path": path,
                "body": text.strip(),
                "blocks": self._split_blocks(body_lines, path, node_id),
                "content_type": "preface",
                "node_id": node_id,
                "parent_node_id": "",
                "title": "文档内容",
                "source_start_line": 1,
                "source_end_line": len(lines),
            })
        return sections

    def _level_to_content_type(self, level: int) -> str:
        if level <= LEVEL_CHAPTER:
            return "chapter"
        if level <= LEVEL_SECTION:
            return "section"
        if level <= LEVEL_SUBSECTION:
            return "subsection"
        return "subsubsection"

    def _split_blocks(self, numbered_lines: list[tuple[int, str]], path: list[str], section_id: str) -> list[dict[str, Any]]:
        raw_blocks = self._collect_raw_blocks(numbered_lines)
        merged_blocks = self._merge_continuation_blocks(raw_blocks)
        blocks = []
        for block in merged_blocks:
            blocks.append({
                "parent_node_id": section_id,
                "block_type": block["block_type"],
                "text": block["text"],
                "title_path": " / ".join(path),
                "title": path[-1] if path else "文档内容",
                "source_start_line": block["source_start_line"],
                "source_end_line": block["source_end_line"],
                "merged_from": block["merged_from"],
            })
        return self._merge_adjacent_special_blocks(blocks)

    @staticmethod
    def _merge_adjacent_special_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merge_types = {"math", "table", "image"}
        merged: list[dict[str, Any]] = []
        for block in blocks:
            bt = block["block_type"]
            if merged and bt in merge_types and merged[-1]["block_type"] == bt:
                prev = merged[-1]
                prev["text"] += "\n\n" + block["text"]
                prev["source_end_line"] = block["source_end_line"]
                prev["merged_from"] = prev.get("merged_from", []) + block.get("merged_from", [])
            else:
                merged.append(block)
        return merged

    def _collect_raw_blocks(self, numbered_lines: list[tuple[int, str]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        buffer: list[tuple[int, str]] = []
        fence_marker: str | None = None
        in_math_block = False
        table_mode = False

        def flush_buffer(block_type: str = "paragraph") -> None:
            nonlocal buffer
            text = self._clean_body([line for _, line in buffer])
            if text:
                line_numbers = [line_no for line_no, _ in buffer]
                blocks.append({
                    "block_type": block_type,
                    "text": text,
                    "source_start_line": min(line_numbers),
                    "source_end_line": max(line_numbers),
                    "merged_from": [[min(line_numbers), max(line_numbers)]],
                })
            buffer = []

        for line_no, line in numbered_lines:
            stripped = line.strip()
            lstripped = line.lstrip()

            if lstripped.startswith("```") or lstripped.startswith("~~~"):
                marker = lstripped[:3]
                buffer.append((line_no, line))
                if fence_marker is None:
                    fence_marker = marker
                elif marker == fence_marker:
                    fence_marker = None
                    flush_buffer("code")
                continue

            if fence_marker is not None:
                buffer.append((line_no, line))
                continue

            if stripped == "$$":
                buffer.append((line_no, line))
                if in_math_block:
                    in_math_block = False
                    flush_buffer("math")
                else:
                    in_math_block = True
                continue

            if in_math_block:
                buffer.append((line_no, line))
                continue

            is_table_line = "|" in line and stripped != ""
            if is_table_line:
                if not table_mode and buffer:
                    flush_buffer("paragraph")
                table_mode = True
                buffer.append((line_no, line))
                continue

            if table_mode:
                flush_buffer("table")
                table_mode = False

            if stripped == "":
                flush_buffer("paragraph")
                continue

            block_type = "image" if re.search(r"!\[[^\]]*\]\([^\)]+\)", line) else "paragraph"
            if block_type == "image" and buffer:
                flush_buffer("paragraph")
            buffer.append((line_no, line))
            if block_type == "image":
                flush_buffer("image")

        if table_mode:
            flush_buffer("table")
        else:
            flush_buffer("paragraph")
        return blocks

    def _merge_continuation_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for block in blocks:
            if merged and self._should_merge_blocks(merged[-1], block):
                previous = merged[-1]
                previous["text"] = self._join_paragraphs(previous["text"], block["text"])
                previous["source_end_line"] = block["source_end_line"]
                previous["merged_from"].extend(block["merged_from"])
                continue
            merged.append({**block, "merged_from": list(block["merged_from"])})
        return merged

    def _should_merge_blocks(self, previous: dict[str, Any], current: dict[str, Any]) -> bool:
        if previous["block_type"] != "paragraph" or current["block_type"] != "paragraph":
            return False
        previous_text = previous["text"].rstrip()
        current_text = current["text"].lstrip()
        if not previous_text or not current_text:
            return False
        if current_text.startswith(("#", "|", "```", "~~~", "![]")):
            return False
        if previous_text[-1] in "。！？；：.!?;:)]）】》”’\"'":
            return False
        if current_text[0] in "，。！？；：,.!?;:)]）】》”’\"'":
            return True
        if re.match(r"^[a-z0-9]", current_text):
            return True
        if re.match(r"^[\u4e00-\u9fff]", current_text):
            return True
        return False

    def _join_paragraphs(self, previous: str, current: str) -> str:
        if current[:1] in "，。！？；：,.!?;:":
            return previous.rstrip() + current.lstrip()
        return previous.rstrip() + current.lstrip()

    def _build_outline(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        roots: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        seen: set[str] = set()
        for section in sections:
            node_id = section["node_id"]
            if node_id in seen:
                continue
            seen.add(node_id)
            node = {
                "node_id": node_id,
                "title": section["title"],
                "logical_level": section["logical_level"],
                "title_path": " / ".join(section["path"]),
                "children": [],
            }
            while stack and stack[-1]["logical_level"] >= node["logical_level"]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                roots.append(node)
            stack.append(node)
        return roots

    def _clean_body(self, lines: list[str]) -> str:
        return "\n".join(lines).strip()

    def _build_node_id(self, index: int, path: list[str]) -> str:
        slug = self._slugify_path(path)
        return f"node_{index}_{slug}" if index >= 0 else f"node_{slug}"

    def _slugify_path(self, path: list[str]) -> str:
        raw = "_".join(path) or "document"
        slug = re.sub(r"\s+", "_", raw)
        slug = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", slug)
        return slug[:120] or "document"


if __name__ == "__main__":
    loader = MarkdownLoader()
    docs = loader.load(r"data\算法基础\算法基础.md")
    print(f"总section数: {len(docs)}")
    for i, d in enumerate(docs[:25]):
        ll = d.get("logical_level")
        tp = d.get("title_path")
        sl = d.get("source_start_line")
        el = d.get("source_end_line")
        ct = d.get("content_type")
        print(f"  [{i}] L{ll} {ct} | {tp} | 行{sl}-{el}")
    print("...")
    for i, d in enumerate(docs):
        ll = d.get("logical_level")
        if ll == 1 and "目录" not in str(d.get("title_path", "")):
            body_preview = str(d.get("text", ""))[:120].replace("\n", " ")
            print(f"\n  L1示例: {d.get('title_path')} | {body_preview}...")