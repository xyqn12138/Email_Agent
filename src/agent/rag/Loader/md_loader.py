import re

from agent.rag.Loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    document_type = "Markdown"
    supported_suffixes = (".md", ".markdown")

    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max(1, max_chunk_size)

    def load(self, file_path: str):
        absolute_path = self.validate_path(file_path)
        text = absolute_path.read_text(encoding="utf-8")
        sections = self._parse_sections(text)
        results = []

        for section_index, section in enumerate(sections):
            results.append(
                self.build_metadata(
                    absolute_path,
                    text=section["body"],
                    heading_level=section["level"],
                    heading_path=" / ".join(section["path"]),
                    heading_titles=section["path"],
                    section_index=section_index,
                    section_type=section["section_type"],
                    blocks=section["blocks"],
                )
            )
        return results

    def _parse_sections(self, text: str) -> list[dict]:
        lines = text.splitlines()
        sections: list[dict] = []
        current_h1: dict | None = None
        current_h2: dict | None = None
        preface_lines: list[str] = []
        fence_marker: str | None = None

        def build_section(level: int, path: list[str], body_lines: list[str], section_type: str) -> None:
            body = self._clean_body(body_lines)
            if not body:
                return
            sections.append(
                {
                    "level": level,
                    "path": path,
                    "body": body,
                    "blocks": self._split_blocks(body),
                    "section_type": section_type,
                }
            )

        def flush_h2() -> None:
            nonlocal current_h2
            if current_h2 is None:
                return
            build_section(2, [current_h2["parent_title"], current_h2["title"]], current_h2["lines"], "h2")
            current_h2 = None

        def flush_h1() -> None:
            nonlocal current_h1
            if current_h1 is None:
                return
            build_section(1, [current_h1["title"]], current_h1["lines"], "h1")
            current_h1 = None

        def flush_preface() -> None:
            nonlocal preface_lines
            build_section(0, ["文档内容"], preface_lines, "preface")
            preface_lines = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if fence_marker is None:
                    fence_marker = marker
                elif marker == fence_marker:
                    fence_marker = None

            if fence_marker is None and re.match(r"^#\s+", stripped):
                flush_h2()
                flush_h1()
                flush_preface()
                current_h1 = {"title": stripped[2:].strip(), "lines": []}
                continue

            if fence_marker is None and re.match(r"^##\s+", stripped):
                flush_h2()
                if current_h1 is None:
                    flush_preface()
                    current_h1 = {"title": "未命名章节", "lines": []}
                current_h2 = {
                    "title": stripped[3:].strip(),
                    "parent_title": current_h1["title"],
                    "lines": [],
                }
                continue

            target = current_h2 or current_h1
            if target is not None:
                target["lines"].append(line)
            else:
                preface_lines.append(line)

        flush_h2()
        flush_h1()
        flush_preface()

        if not sections and text.strip():
            body = text.strip()
            sections.append(
                {
                    "level": 0,
                    "path": ["文档内容"],
                    "body": body,
                    "blocks": self._split_blocks(body),
                    "section_type": "preface",
                }
            )
        return sections

    def _clean_body(self, lines: list[str]) -> str:
        return "\n".join(lines).strip()

    def _split_blocks(self, body: str) -> list[dict]:
        lines = body.splitlines()
        blocks: list[dict] = []
        buffer: list[str] = []
        fence_marker: str | None = None
        in_math_block = False
        table_mode = False

        def flush_buffer(block_type: str = "paragraph") -> None:
            nonlocal buffer
            text = "\n".join(buffer).strip()
            if text:
                blocks.append({"block_type": block_type, "text": text})
            buffer = []

        for line in lines:
            stripped = line.strip()
            lstripped = line.lstrip()

            if lstripped.startswith("```") or lstripped.startswith("~~~"):
                marker = lstripped[:3]
                buffer.append(line)
                if fence_marker is None:
                    fence_marker = marker
                elif marker == fence_marker:
                    fence_marker = None
                    flush_buffer("code")
                continue

            if fence_marker is not None:
                buffer.append(line)
                continue

            if stripped == "$$":
                buffer.append(line)
                if in_math_block:
                    in_math_block = False
                    flush_buffer("math")
                else:
                    in_math_block = True
                continue

            if in_math_block:
                buffer.append(line)
                continue

            is_table_line = "|" in line and stripped != ""
            if is_table_line:
                if not table_mode and buffer:
                    flush_buffer("paragraph")
                table_mode = True
                buffer.append(line)
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
            buffer.append(line)
            if block_type == "image":
                flush_buffer("image")

        if table_mode:
            flush_buffer("table")
        else:
            flush_buffer("paragraph")
        return blocks


if __name__ == "__main__":
    loader = MarkdownLoader()
    docs = loader.load(r"src\agent\data\minerU.md")
    if docs:
        # for doc in docs[0:5]:
        #     print(doc["heading_path"])
        #     print(doc["text"][:200])
        #     print("-----------------")
        print(docs[10])
