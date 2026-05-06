from pathlib import Path

from agent.rag.data_embedding import build_default_pipeline_registry


def test_markdown_pipeline_prepares_structured_data(tmp_path: Path) -> None:
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text(
        "\n".join(
            [
                "# 总览",
                "这是一级标题下的说明。",
                "",
                "| 列1 | 列2 |",
                "| --- | --- |",
                "| A   | B   |",
                "",
                "## 细节",
                "这里是二级标题的正文。",
                "",
                "```python",
                "print('hello')",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    pipeline = build_default_pipeline_registry().resolve(str(markdown_file))
    prepared = pipeline.prepare(str(markdown_file))

    assert prepared.pipeline_name == "markdown"
    assert prepared.metadata["filename"] == "sample.md"
    assert len(prepared.loaded_content) == 2

    first_section = prepared.loaded_content[0]
    assert first_section["heading_titles"] == ["总览"]
    assert [block["block_type"] for block in first_section["blocks"]] == ["paragraph", "table"]

    level1_chunk = next(chunk for chunk in prepared.chunks if chunk["chunk_level"] == 1)
    assert level1_chunk["text"].startswith("# 总览")

    level2_chunk = next(chunk for chunk in prepared.chunks if chunk["chunk_level"] == 2)
    assert level2_chunk["heading_path"] == "总览 / 细节"

    assert any(
        chunk["chunk_level"] == 3 and chunk["block_type"] == "table"
        for chunk in prepared.chunks
    )
    assert any(
        chunk["chunk_level"] == 3 and chunk["block_type"] == "code"
        for chunk in prepared.chunks
    )
