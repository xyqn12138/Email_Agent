from pathlib import Path
import unittest

from agent.rag.data_embedding import build_default_pipeline_registry


def test_markdown_pipeline_prepares_structured_data(tmp_path: Path) -> None:
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text(
        "\n".join(
            [
                "# 第1章 示例",
                "这是一级标题下的说明。",
                "",
                "| 列1 | 列2 |",
                "| --- | --- |",
                "| A   | B   |",
                "",
                "# 1.1 二级标题",
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
    assert first_section["title"] == "第1章 示例"
    assert first_section["logical_level"] == 1
    assert first_section["content_type"] == "chapter"
    assert first_section["node_id"]
    assert [block["block_type"] for block in first_section["blocks"]] == ["paragraph", "table"]
    assert all(block["block_id"] for block in first_section["blocks"])

    second_section = prepared.loaded_content[1]
    assert second_section["logical_level"] == 2
    assert second_section["content_type"] == "section"
    assert second_section["title"] == "1.1 二级标题"

    l1_chunks = [chunk for chunk in prepared.chunks if chunk["chunk_level"] == 1]
    l2_chunks = [chunk for chunk in prepared.chunks if chunk["chunk_level"] == 2]
    l3_chunks = [chunk for chunk in prepared.chunks if chunk["chunk_level"] == 3]
    l4_chunks = [chunk for chunk in prepared.chunks if chunk["chunk_level"] == 4]

    assert len(l1_chunks) >= 1
    assert len(l2_chunks) >= 1
    assert len(l3_chunks) >= 1

    l1_chunk = l1_chunks[0]
    assert l1_chunk["chunk_id"].endswith("_L1_0")
    assert l1_chunk["parent_chunk_id"] == ""
    assert l1_chunk["root_chunk_id"] == l1_chunk["chunk_id"]

    l2_chunk = l2_chunks[0]
    assert "_L2_" in l2_chunk["chunk_id"]
    assert l2_chunk["parent_chunk_id"] == l1_chunk["chunk_id"]
    assert l2_chunk["root_chunk_id"] == l1_chunk["chunk_id"]

    l3_code = next((c for c in l3_chunks if "print('hello')" in c["text"]), None)
    assert l3_code is not None
    assert l3_code["parent_chunk_id"] == l2_chunk["chunk_id"]

    l3_table = next((c for c in l3_chunks if "列1" in c["text"]), None)
    assert l3_table is not None


def test_markdown_loader_merges_cross_page_paragraphs(tmp_path: Path) -> None:
    markdown_file = tmp_path / "cross_page.md"
    markdown_file.write_text(
        "\n".join(
            [
                "# 第0章",
                "## 0.2.2 存储器",
                "存储器是计算机系统中的记忆设备，用来存放程序和数据。理想情形下，存储器应该具备大容量、高速度、低价位（价格/位，简称位价）的特性，但是大容量和高速度是相互制约的，所以",
                "",
                "单个存储部件很难同时满足这三个特性。为最大化体现这三个特性，存储器系统采用了分层结构。",
            ]
        ),
        encoding="utf-8",
    )

    pipeline = build_default_pipeline_registry().resolve(str(markdown_file))
    prepared = pipeline.prepare(str(markdown_file))

    section = prepared.loaded_content[0]
    paragraph_blocks = [block for block in section["blocks"] if block["block_type"] == "paragraph"]
    assert len(paragraph_blocks) == 1
    assert "所以单个存储部件" in paragraph_blocks[0]["text"]
    assert paragraph_blocks[0]["merged_from"] == [[3, 3], [5, 5]]


def test_four_layer_split_structure(tmp_path: Path) -> None:
    markdown_file = tmp_path / "four_layer.md"
    markdown_file.write_text(
        "\n".join(
            [
                "# 第2章 递归与分治法",
                "分治法的基本思想是将一个规模为n的问题分解为k个规模较小的子问题。",
                "",
                "# 2.1 基本思想",
                "分治法的基本思想如下：将一个难以直接解决的大问题，分割成一些规模较小的相同问题，以便各个击破，分而治之。",
                "这些子问题互相独立且与原问题相同。",
                "",
                "分治法有三个步骤：",
                "第一步：分解，将原问题分解为若干个规模较小的子问题。",
                "第二步：求解，递归地求解各子问题。",
                "第三步：合并，将各子问题的解合并为原问题的解。",
            ]
        ),
        encoding="utf-8",
    )

    pipeline = build_default_pipeline_registry().resolve(str(markdown_file))
    prepared = pipeline.prepare(str(markdown_file))

    levels = sorted(set(c["chunk_level"] for c in prepared.chunks))
    assert 1 in levels
    assert 2 in levels
    assert 3 in levels

    l1_chunks = [c for c in prepared.chunks if c["chunk_level"] == 1]
    l2_chunks = [c for c in prepared.chunks if c["chunk_level"] == 2]
    l3_chunks = [c for c in prepared.chunks if c["chunk_level"] == 3]

    assert len(l1_chunks) == 1
    assert "递归与分治法" in l1_chunks[0]["text"]
    assert len(l2_chunks) >= 1
    assert len(l3_chunks) >= 2

    for l3 in l3_chunks:
        assert l3["parent_chunk_id"] in [c["chunk_id"] for c in l2_chunks] or \
               l3["parent_chunk_id"] == l1_chunks[0]["chunk_id"]

if __name__ == "__main__":
    unittest.main()