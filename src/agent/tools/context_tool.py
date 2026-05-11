from langchain_core.tools import tool
from agent.rag.milvus_manage import MilvusManage
from agent.utils.logger_handler import get_logger

logger = get_logger()

_milvus: MilvusManage | None = None


def _get_milvus() -> MilvusManage:
    global _milvus
    if _milvus is None:
        _milvus = MilvusManage()
    return _milvus


def _format_chunks(chunks: list[dict], label: str) -> list[str]:
    parts: list[str] = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        title = c.get("title_path", "")
        text = c.get("text", "")
        page = c.get("page_number", 0)
        image_paths = c.get("image_paths", "")
        image_hint = f"\n  包含图片: {image_paths}" if image_paths else ""
        parts.append(f"[{label}] chunk_id={cid} | {title} (p.{page}){image_hint}\n{text}")
    return parts


@tool("fetch_neighbor_context")
def fetch_neighbor_context(
    chunk_id: str,
    n_before: int = 2,
    n_after: int = 2,
) -> str:
    """
    获取指定文档片段的前序和后续片段，用于阅读被截断内容的上下文。
    当 knowledge_base_search 返回的结果内容不完整（如公式被截断、代码缺少结尾等）时，
    使用此工具获取该片段前后的连续内容。

    Args:
        chunk_id: 文档片段的唯一标识符，从 knowledge_base_search 返回结果中获取。
            格式如 "a1b2c3d4_L3_0012"。
        n_before: 向前取几个片段，默认2。
        n_after: 向后取几个片段，默认2。

    Returns:
        指定片段的前序和后续内容，按顺序排列。
    """
    milvus = _get_milvus()
    neighbors = milvus.get_neighbor_chunks(chunk_id, n_before=n_before, n_after=n_after)

    before = neighbors.get("before", [])
    after = neighbors.get("after", [])

    if not before and not after:
        return f"未找到 chunk_id={chunk_id} 的相邻片段，可能是该片段位于章节首尾。"

    lines: list[str] = []
    if before:
        before.reverse()
        lines.extend(_format_chunks(before, "前序"))

    current = milvus.query_by_chunk_ids(
        [chunk_id],
        output_fields=["chunk_id", "text", "title_path", "page_number", "image_paths"],
    )
    if current:
        lines.extend(_format_chunks(current, "当前"))

    if after:
        lines.extend(_format_chunks(after, "后续"))

    return "\n\n---\n\n".join(lines)
