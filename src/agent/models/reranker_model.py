import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from agent.utils.logger_handler import get_logger

load_dotenv()

logger = get_logger()

DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
DEFAULT_RERANK_BASE_URL = os.getenv(
    "RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1"
)
DEFAULT_RERANK_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    document: str


class RerankerModel:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or DEFAULT_RERANK_MODEL
        self.client = OpenAI(
            api_key=api_key or DEFAULT_RERANK_API_KEY,
            base_url=base_url or DEFAULT_RERANK_BASE_URL,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        top_n = top_n or DEFAULT_RERANK_TOP_N
        top_n = min(top_n, len(documents))

        try:
            response = self.client.post(
                "/reranks",
                body={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                cast_to=object,
            )
        except Exception as exc:
            logger.error(f"Rerank request failed: {exc}")
            return []

        results_data = response if isinstance(response, list) else response.get("results", [])
        rerank_results: list[RerankResult] = []
        for item in results_data:
            if isinstance(item, dict):
                rerank_results.append(
                    RerankResult(
                        index=item.get("index", 0),
                        relevance_score=item.get("relevance_score", 0.0),
                        document=item.get("document", ""),
                    )
                )
        rerank_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return rerank_results


if __name__ == "__main__":
    reranker = RerankerModel()
    results = reranker.rerank(
        query="什么是重排序模型",
        documents=[
            "重排序模型广泛应用于搜索引擎和推荐系统，按相关性对候选文本进行排序",
            "量子计算是计算科学的前沿领域",
            "预训练语言模型的发展为重排序模型带来了新的进展",
        ],
        top_n=2,
    )
    for r in results:
        print(f"[{r.index}] score={r.relevance_score:.4f} | {r.document[:60]}")
