from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings


class MilvusVectorStore:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedding = DashScopeEmbeddings(
            model="text-embedding-v4"
        )

        self.vector_store = Milvus(
            embedding_function=self.embedding,
            collection_name=collection_name,

            # hybrid search
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],

            # index 参数
            index_params=[
                {
                    "field_name": "dense",
                    "index_name": "dense_index",
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 64},
                },
                {
                    "field_name": "sparse",
                    "index_name": "sparse_index",
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        "bm25_k1": 1.2,
                        "bm25_b": 0.75,
                    },
                },
            ],

            connection_args={
                "uri": "http://127.0.0.1:19530"
            },

            consistency_level="session",

            auto_id=True,
        )

    # 写入
    def add_documents(self, documents: list[Document]):
        return self.vector_store.add_documents(documents)

    # 相似度检索
    def similarity_search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

    # 带分数（调试/评估很有用）
    def similarity_search_with_score(self, query: str, k: int = 5):
        return self.vector_store.similarity_search_with_score(query, k=k)

    # 删除（可选）
    def delete(self, ids: list[str]):
        return self.vector_store.delete(ids=ids)
    