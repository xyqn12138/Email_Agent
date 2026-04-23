from typing import List, Dict, Any
from agent.models.chat_model import chat_model
from agent.models.embedding_model import EmbeddingModel
from agent.rag.milvus_manage import MilvusManage
from agent.utils.logger_handler import get_logger
from pymilvus import AnnSearchRequest, RRFRanker
import os

logger = get_logger()

class Retriever:
    def __init__(self, model_name: str = "dashscope", dimensions: int = 1024):
        self.embedding_model = EmbeddingModel(model_name=model_name, dimensions=dimensions)
        self.milvus_manager = MilvusManage()
        self.chat_model = chat_model
        self.collection_name = os.getenv('COLLECTION_NAME')

    def rewrite_query(self, query: str) -> str:
        """
        Query rewriting using LLM.
        """
        prompt = f"你是一个搜索专家。请将以下用户查询改写为更适合在文档库中进行语义检索的表达方式，保持原意，但增加可能的关键词。直接输出改写后的查询：\n\n查询：{query}"
        response = self.chat_model.invoke(prompt)
        rewritten_query = response.content.strip()
        logger.info(f"Original query: {query} -> Rewritten: {rewritten_query}")
        return rewritten_query

    def hybrid_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Hybrid search (Dense + BM25) on layers 2 and 3.
        """
        client = self.milvus_manager._get_connect()
        
        # 1. Generate embeddings
        dense_vec = self.embedding_model.embed_query(query)
        
        # 2. Prepare search requests for Layers 2 and 3
        # Search Layer 2 & 3
        search_filter = "chunk_level in [2, 3]"
        
        # Dense search request
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="text_dense",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=search_filter
        )
        
        # Sparse (BM25) search request
        # Note: In Milvus 2.4+, BM25 search uses a different approach if using Function
        # Here we assume the client.hybrid_search will handle it or we use two separate searches
        
        # Sparse search request (using BM25)
        sparse_req = AnnSearchRequest(
            data=[query], # For BM25, we pass the text query
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
            expr=search_filter
        )

        # 3. Hybrid search with RRF
        res = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["text", "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "filename"]
        )
        
        # Flatten results
        results = []
        if res and len(res) > 0:
            for hit in res[0]:
                results.append({
                    "text": hit.get("text"),
                    "chunk_id": hit.get("chunk_id"),
                    "parent_chunk_id": hit.get("parent_chunk_id"),
                    "root_chunk_id": hit.get("root_chunk_id"),
                    "chunk_level": hit.get("chunk_level"),
                    "filename": hit.get("filename"),
                    "score": hit.score
                })
        
        return results

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using LLM as a placeholder for a dedicated reranker.
        """
        if not results:
            return []
        
        # Simple LLM reranking (can be replaced by BGE-Reranker etc.)
        logger.info(f"Reranking {len(results)} results...")
        
        # To keep it efficient, we just return for now or implement a simple score-based logic
        # In a real scenario, you'd use a dedicated reranker model
        return results

    def fetch_multi_layer_context(self, search_results: List[Dict]) -> List[Dict]:
        """
        Recall Layers 1, 2, 3 context for the search results.
        L1: Chunk1, L2: Chunk2, L3: Chunk3
        """
        client = self.milvus_manager._get_connect()
        final_contexts = []
        
        processed_ids = set()

        for hit in search_results:
            chunk_id = hit["chunk_id"]
            if chunk_id in processed_ids:
                continue
            
            context = {
                "search_hit": hit["text"],
                "level": hit["chunk_level"],
                "filename": hit["filename"],
                "chunk1_text": "",
                "chunk2_text": "",
                "chunk3_text": hit["text"] if hit["chunk_level"] == 3 else ""
            }

            # Fetch L1 (Chunk1) and L2 (Chunk2) context
            ids_to_fetch = []
            if hit["root_chunk_id"]:
                ids_to_fetch.append(hit["root_chunk_id"])
            if hit["parent_chunk_id"] and hit["parent_chunk_id"] != hit["root_chunk_id"]:
                ids_to_fetch.append(hit["parent_chunk_id"])

            if ids_to_fetch:
                # Milvus query to get parent/root texts
                filter_expr = f"chunk_id in {ids_to_fetch}"
                ancestors = client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=["text", "chunk_id", "chunk_level"]
                )
                
                for a in ancestors:
                    if a["chunk_level"] == 1:
                        context["chunk1_text"] = a["text"]
                    elif a["chunk_level"] == 2:
                        context["chunk2_text"] = a["text"]

            # Special case: if hit was L2, chunk2_text should be the hit
            if hit["chunk_level"] == 2:
                context["chunk2_text"] = hit["text"]

            final_contexts.append(context)
            processed_ids.add(chunk_id)
            
        return final_contexts

    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Complete retrieval pipeline.
        """
        # 1. Query Rewriting
        rewritten_query = self.rewrite_query(query)
        
        # 2. Hybrid Search on L2 & L3
        search_results = self.hybrid_search(rewritten_query, limit=limit)
        
        # 3. Rerank
        reranked_results = self.rerank(rewritten_query, search_results)
        
        # 4. Multi-layer Recall
        final_contexts = self.fetch_multi_layer_context(reranked_results)
        
        return final_contexts

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("这篇文章提出了什么新算法？")
    for r in results:
        print(f"File: {r['filename']}, Level: {r['level']}")
        print(f"Chunk1: {r['chunk1_text'][:50]}...")
        print(f"Chunk2: {r['chunk2_text'][:50]}...")
        print(f"Search Hit: {r['search_hit'][:50]}...")
        print("-" * 20)
