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
        Hybrid search (Dense + BM25) on layers 3 and 4 (main recall layers).
        """
        client = self.milvus_manager._get_connect()
        
        dense_vec = self.embedding_model.embed_queries(query)[0]
        
        search_filter = "chunk_level in [3, 4]"
        
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="text_dense",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=search_filter
        )
        
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
            expr=search_filter
        )

        res = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["text", "chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level", "filename", "title_path", "content_type", "page_number"]
        )
        
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
                    "title_path": hit.get("title_path", ""),
                    "content_type": hit.get("content_type", ""),
                    "page_number": hit.get("page_number", 0),
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
        Recall Layers 1, 2, 3, 4 context for the search results.
        L1: 章, L2: 小节, L3: 知识块, L4: 细粒度证据块
        """
        final_contexts = []
        processed_ids = set()

        all_ancestor_ids: set[str] = set()
        for hit in search_results:
            cid = hit["chunk_id"]
            if cid in processed_ids:
                continue
            processed_ids.add(cid)
            if hit.get("root_chunk_id"):
                all_ancestor_ids.add(hit["root_chunk_id"])
            if hit.get("parent_chunk_id"):
                all_ancestor_ids.add(hit["parent_chunk_id"])

        ancestor_map: dict[str, dict] = {}
        if all_ancestor_ids:
            ancestor_list = list(all_ancestor_ids)
            for i in range(0, len(ancestor_list), 100):
                batch = ancestor_list[i : i + 100]
                rows = self.milvus_manager.query_by_chunk_ids(
                    batch, output_fields=["text", "chunk_id", "chunk_level", "title_path"]
                )
                for row in rows:
                    ancestor_map[row["chunk_id"]] = row

            missing_l2_ids: set[str] = set()
            for hit in search_results:
                if hit["chunk_level"] == 4 and hit.get("parent_chunk_id"):
                    parent = ancestor_map.get(hit["parent_chunk_id"])
                    if parent and parent.get("chunk_level") == 3 and parent.get("parent_chunk_id"):
                        missing_l2_ids.add(parent["parent_chunk_id"])
            missing_l2_ids -= ancestor_map.keys()
            if missing_l2_ids:
                batch = list(missing_l2_ids)
                for i in range(0, len(batch), 100):
                    rows = self.milvus_manager.query_by_chunk_ids(
                        batch[i : i + 100], output_fields=["text", "chunk_id", "chunk_level", "title_path"]
                    )
                    for row in rows:
                        ancestor_map[row["chunk_id"]] = row

        processed_ids.clear()
        for hit in search_results:
            chunk_id = hit["chunk_id"]
            if chunk_id in processed_ids:
                continue

            context = {
                "search_hit": hit["text"],
                "level": hit["chunk_level"],
                "filename": hit["filename"],
                "title_path": hit.get("title_path", ""),
                "content_type": hit.get("content_type", ""),
                "page_number": hit.get("page_number", 0),
                "chunk1_text": "",
                "chunk2_text": "",
                "chunk3_text": hit["text"] if hit["chunk_level"] == 3 else "",
                "chunk4_text": hit["text"] if hit["chunk_level"] == 4 else "",
            }

            ids_to_check: set[str] = set()
            if hit.get("root_chunk_id"):
                ids_to_check.add(hit["root_chunk_id"])
            if hit.get("parent_chunk_id"):
                ids_to_check.add(hit["parent_chunk_id"])
            if hit["chunk_level"] == 4 and hit.get("parent_chunk_id"):
                parent_node = ancestor_map.get(hit["parent_chunk_id"])
                if parent_node and parent_node.get("parent_chunk_id"):
                    ids_to_check.add(parent_node["parent_chunk_id"])

            for aid in ids_to_check:
                node = ancestor_map.get(aid)
                if not node:
                    continue
                level = node["chunk_level"]
                if level == 1:
                    context["chunk1_text"] = node["text"]
                elif level == 2:
                    context["chunk2_text"] = node["text"]
                elif level == 3:
                    context["chunk3_text"] = node["text"]

            if hit["chunk_level"] == 3:
                context["chunk3_text"] = hit["text"]
            elif hit["chunk_level"] == 2:
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
        
        # 2. Hybrid Search on L3 & L4
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
