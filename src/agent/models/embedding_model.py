import os
from dataclasses import dataclass
from typing import Protocol

import requests
from dotenv import load_dotenv
from requests import RequestException

load_dotenv()

DEFAULT_ZHIPU_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
DEFAULT_DASHSCOPE_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
DEFAULT_LOCAL_DOCUMENT_MODEL = os.getenv(
    "LOCAL_DOCUMENT_EMBEDDING_MODEL",
    "qwen3-embedding-0.6b",
)
DEFAULT_LOCAL_QUERY_MODEL = os.getenv(
    "LOCAL_QUERY_EMBEDDING_MODEL",
    "qwen3-embedding-4b",
)
DEFAULT_LOCAL_BASE_URL = os.getenv(
    "RAG_SERVICE_URL",
    "http://127.0.0.1:8000"
)
DEFAULT_TIMEOUT = float(os.getenv("EMBEDDING_API_TIMEOUT", "120"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
DEFAULT_MAX_TEXT_CHARS = int(os.getenv("EMBEDDING_MAX_TEXT_CHARS", "6000"))
DEFAULT_QUERY_INSTRUCT = os.getenv(
    "EMBEDDING_QUERY_INSTRUCT",
    "Given a web search query, retrieve relevant passages that answer the query",
)
DEFAULT_DOCUMENT_INSTRUCT = os.getenv(
    "EMBEDDING_DOCUMENT_INSTRUCT",
    "Given a passage, retrieve semantically similar passages for knowledge base retrieval",
)


@dataclass(frozen=True)
class EmbeddingProfile:
    model: str
    instruct: str = ""


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], *, profile: EmbeddingProfile, dimensions: int) -> list[list[float]]:
        ...


class BaseHTTPEmbeddingProvider:
    def __init__(self, *, timeout: float, batch_size: int):
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def _chunk_texts(self, texts: list[str]) -> list[list[str]]:
        return [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

    def _parse_embeddings_response(self, data: dict) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise ValueError("Invalid embedding response: missing data field")
        embeddings: list[list[float]] = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("Invalid embedding response: malformed embedding item")
            embeddings.append(item["embedding"])
        return embeddings

    def _post_embeddings(self, *, url: str, headers: dict[str, str], payload: dict) -> list[list[float]]:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if not response.ok:
                raise RuntimeError(
                    f"Embedding request failed: {response.status_code} {response.reason} | "
                    f"url={url} | body={response.text[:500]}"
                )
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("Embedding response is not valid JSON") from exc

        return self._parse_embeddings_response(data)


class ZhipuEmbeddingProvider(BaseHTTPEmbeddingProvider):
    def __init__(self, *, timeout: float, batch_size: int):
        super().__init__(timeout=timeout, batch_size=batch_size)
        self.url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.api_key = os.getenv("ZHIPUAI_API_KEY")

    def embed(self, texts: list[str], *, profile: EmbeddingProfile, dimensions: int) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        all_embeddings: list[list[float]] = []
        for chunk in self._chunk_texts(texts):
            payload = {
                "model": profile.model,
                "input": chunk if len(chunk) > 1 else chunk[0],
                "dimensions": dimensions,
            }
            embeddings = self._post_embeddings(url=self.url, headers=headers, payload=payload)
            if len(embeddings) != len(chunk):
                raise ValueError("Embedding response size does not match input size")
            all_embeddings.extend(embeddings)
        return all_embeddings


class DashscopeEmbeddingProvider(BaseHTTPEmbeddingProvider):
    def __init__(self, *, timeout: float, batch_size: int):
        super().__init__(timeout=timeout, batch_size=batch_size)
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

    def embed(self, texts: list[str], *, profile: EmbeddingProfile, dimensions: int) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        all_embeddings: list[list[float]] = []
        for chunk in self._chunk_texts(texts):
            payload = {
                "model": profile.model,
                "input": chunk if len(chunk) > 1 else chunk[0],
                "dimensions": dimensions,
                "instruct": profile.instruct,
            }
            embeddings = self._post_embeddings(url=self.url, headers=headers, payload=payload)
            if len(embeddings) != len(chunk):
                raise ValueError("Embedding response size does not match input size")
            all_embeddings.extend(embeddings)
        return all_embeddings


class LocalEmbeddingProvider(BaseHTTPEmbeddingProvider):
    def __init__(self, *, base_url: str, timeout: float, batch_size: int):
        super().__init__(timeout=timeout, batch_size=batch_size)
        self.url = f"{base_url.rstrip('/')}/v1/embeddings"

    def embed(self, texts: list[str], *, profile: EmbeddingProfile, dimensions: int) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        all_embeddings: list[list[float]] = []
        for chunk in self._chunk_texts(texts):
            payload = {
                "model": profile.model,
                "input": chunk if len(chunk) > 1 else chunk[0],
                "dimensions": dimensions,
                "instruct": profile.instruct,
            }
            embeddings = self._post_embeddings(url=self.url, headers=headers, payload=payload)
            if len(embeddings) != len(chunk):
                raise ValueError("Embedding response size does not match input size")
            all_embeddings.extend(embeddings)
        return all_embeddings


class EmbeddingModel:
    def __init__(self, model_name: str = "local", dimensions: int = 1024):
        self.model_name = model_name
        self.dimensions = dimensions
        self.request_timeout = DEFAULT_TIMEOUT
        self.batch_size = max(1, DEFAULT_BATCH_SIZE)
        self.query_profiles = {
            "zhipuai": EmbeddingProfile(model=DEFAULT_ZHIPU_MODEL),
            "dashscope": EmbeddingProfile(model=DEFAULT_DASHSCOPE_MODEL, instruct=DEFAULT_QUERY_INSTRUCT),
            "local": EmbeddingProfile(model=DEFAULT_LOCAL_QUERY_MODEL, instruct=DEFAULT_QUERY_INSTRUCT),
        }
        self.document_profiles = {
            "zhipuai": EmbeddingProfile(model=DEFAULT_ZHIPU_MODEL),
            "dashscope": EmbeddingProfile(model=DEFAULT_DASHSCOPE_MODEL, instruct=DEFAULT_DOCUMENT_INSTRUCT),
            "local": EmbeddingProfile(model=DEFAULT_LOCAL_DOCUMENT_MODEL, instruct=DEFAULT_DOCUMENT_INSTRUCT),
        }
        self.providers: dict[str, EmbeddingProvider] = {
            "zhipuai": ZhipuEmbeddingProvider(timeout=self.request_timeout, batch_size=self.batch_size),
            "dashscope": DashscopeEmbeddingProvider(timeout=self.request_timeout, batch_size=self.batch_size),
            "local": LocalEmbeddingProvider(
                base_url=DEFAULT_LOCAL_BASE_URL,
                timeout=self.request_timeout,
                batch_size=self.batch_size,
            ),
        }

    def _normalize_input(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        return texts

    @staticmethod
    def _truncate_texts(texts: list[str], max_chars: int) -> list[str]:
        truncated = []
        for t in texts:
            if len(t) > max_chars:
                truncated.append(t[:max_chars])
            else:
                truncated.append(t)
        return truncated

    def _get_profile(self, *, is_query: bool) -> EmbeddingProfile:
        profiles = self.query_profiles if is_query else self.document_profiles
        profile = profiles.get(self.model_name)
        if profile is None:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        return profile

    def _get_provider(self) -> EmbeddingProvider:
        provider = self.providers.get(self.model_name)
        if provider is None:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        return provider

    def embed(self, texts: str | list[str], *, is_query: bool) -> list[list[float]]:
        normalized_texts = self._normalize_input(texts)
        if not normalized_texts:
            return []
        normalized_texts = self._truncate_texts(normalized_texts, DEFAULT_MAX_TEXT_CHARS)
        provider = self._get_provider()
        profile = self._get_profile(is_query=is_query)
        return provider.embed(normalized_texts, profile=profile, dimensions=self.dimensions)

    def embed_queries(self, texts: str | list[str]) -> list[list[float]]:
        return self.embed(texts, is_query=True)

    def embed_documents(self, texts: str | list[str]) -> list[list[float]]:
        return self.embed(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_queries(text)
        if not embeddings:
            raise ValueError("Embedding service returned empty result for query")
        return embeddings[0]


if __name__ == "__main__":
    txt = "你好"
    embedding_model = EmbeddingModel(model_name="dashscope", dimensions=1024)
    embedding = embedding_model.embed_query(txt)
    print(f"Embedding for '{txt}': {embedding[:10]}...")
