import os
from typing import Literal

from langchain_core.tools import tool
from agent.rag.retriever import Retriever
from agent.utils.logger_handler import get_logger

logger = get_logger()

RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3"))
ADVANCED_HYBRID_LIMIT = int(os.getenv("ADVANCED_HYBRID_LIMIT", "20"))
ADVANCED_RERANK_TOP_N = int(os.getenv("ADVANCED_RERANK_TOP_N", "5"))

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _format_results(contexts: list[dict]) -> str:
    if not contexts:
        return "知识库中未找到相关内容。"

    parts: list[str] = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title_path", "")
        source = ctx.get("filename", "")
        chunk_id = ctx.get("chunk_id", "")
        text = ctx.get("chunk3_text") or ctx.get("chunk4_text") or ctx.get("search_hit", "")
        rerank_score = ctx.get("rerank_score")
        score_hint = f" [相关度: {rerank_score:.2f}]" if rerank_score else ""
        image_paths = ctx.get("image_paths", "")
        image_hint = f"\n  包含图片: {image_paths}" if image_paths else ""
        parts.append(f"[{i}]{score_hint} {title}（来源: {source}, chunk_id: {chunk_id}）{image_hint}\n{text}")
    return "\n\n---\n\n".join(parts)


def _has_low_quality(results: list[dict]) -> bool:
    if not results:
        return True
    top_score = results[0].get("rerank_score")
    if top_score is None:
        return False
    return top_score < RERANK_SCORE_THRESHOLD


@tool("knowledge_base_search")
def knowledge_base_search(
    query: str,
    mode: Literal["auto", "standard", "hyde"] = "auto",
    limit: int = 3,
    advanced: bool = False,
) -> str:
    """
    从个人知识库中检索与查询相关的文档片段。
    当用户提出与学习资料、教材内容、课程知识相关的问题时，优先使用此工具。

    Args:
        query: 检索查询，描述你想要查找的知识点
        mode: 检索模式。
            - "auto"（默认）: 自动选择最佳路径。短查询走双阶段检索（先定位章节再召回内容），
              长查询直接搜索。若结果相关度低，自动升级为 HyDE 重检索。
            - "standard": 固定走标准混合检索（Dense + BM25），不做 HyDE。
            - "hyde": 固定走 HyDE 模式，先由 LLM 生成假想答案再用其检索，
              适合查询与文档表述差异大的场景。
        limit: 返回结果数量，默认3。当首次结果不佳时，高级重检索会自动增大召回量。
        advanced: 是否开启高级检索模式。开启后使用更大的召回池和更精确的重排序，
            适合首次检索效果不佳或问题较复杂时使用。

    Returns:
        知识库中的相关文档片段，包含章节标题、来源和相关度分数。
    """
    retriever = _get_retriever()

    if advanced:
        contexts = retriever.retrieve(
            query,
            limit=ADVANCED_RERANK_TOP_N,
            use_hyde=True,
        )
        logger.info(f"Advanced retrieval completed, returned {len(contexts)} results")
        return _format_results(contexts)

    use_hyde = mode == "hyde"
    contexts = retriever.retrieve(query, limit=limit, use_hyde=use_hyde)

    if mode == "auto" and not use_hyde and _has_low_quality(contexts):
        logger.info(
            f"Low quality detected (top score={contexts[0].get('rerank_score', 'N/A') if contexts else 'N/A'}), "
            f"upgrading to HyDE re-retrieval"
        )
        contexts = retriever.retrieve(
            query,
            limit=max(limit, 5),
            use_hyde=True,
        )
        logger.info(f"HyDE re-retrieval completed, returned {len(contexts)} results")

    return _format_results(contexts)
