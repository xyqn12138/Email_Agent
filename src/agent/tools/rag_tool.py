from langchain_core.tools import tool
from agent.rag.retriever import Retriever


@tool("knowledge_base_search")
def knowledge_base_search(query: str) -> str:
    """
    从个人知识库中检索与查询相关的文档片段。
    当用户提出与学习资料、教材内容、课程知识相关的问题时，优先使用此工具。

    Args:
        query: 检索查询，描述你想要查找的知识点

    Returns:
        知识库中的相关文档片段，包含章节标题和正文内容
    """
    retriever = Retriever()
    contexts = retriever.retrieve(query, limit=3)
    if not contexts:
        return "知识库中未找到相关内容。"

    parts: list[str] = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title_path", "")
        source = ctx.get("filename", "")
        text = ctx.get("chunk3_text") or ctx.get("chunk4_text") or ctx.get("search_hit", "")
        parts.append(
            f"[{i}] {title}（来源: {source}）\n{text}"
        )
    return "\n\n---\n\n".join(parts)
