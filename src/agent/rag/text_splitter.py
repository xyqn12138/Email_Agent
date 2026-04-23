from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

class ThreeLayerSplitter:
    def __init__(
        self, 
        l1_size: int = 1200, l1_overlap: int = 150,
        l2_size: int = 600, l2_overlap: int = 100,
        l3_size: int = 300, l3_overlap: int = 50
    ):
        # Layer 1 Splitter (Large Chunks)
        self.l1_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l1_size,
            chunk_overlap=l1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        # Layer 2 Splitter (Medium Chunks)
        self.l2_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l2_size,
            chunk_overlap=l2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        # Layer 3 Splitter (Small Chunks)
        self.l3_splitter = RecursiveCharacterTextSplitter(
            chunk_size=l3_size,
            chunk_overlap=l3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into three layers hierarchically using RecursiveCharacterTextSplitter:
        FullText -> Chunk1 (L1) -> Chunk2 (L2) -> Chunk3 (L3)
        """
        root_doc_id = str(uuid.uuid4())
        results = []

        # Layer 1: Large Chunks (Chunk1)
        layer1_chunks = self.l1_splitter.split_text(text)
        
        for l1_idx, l1_text in enumerate(layer1_chunks):
            l1_id = f"{root_doc_id}_l1_{l1_idx}"
            results.append({
                "text": l1_text,
                "chunk_id": l1_id,
                "parent_chunk_id": "", # Use empty string instead of None for Milvus
                "root_chunk_id": l1_id,
                "chunk_level": 1,
                **metadata
            })

            # Layer 2: Medium Chunks (Chunk2) - Split from L1
            layer2_chunks = self.l2_splitter.split_text(l1_text)
            for l2_idx, l2_text in enumerate(layer2_chunks):
                l2_id = f"{l1_id}_l2_{l2_idx}"
                results.append({
                    "text": l2_text,
                    "chunk_id": l2_id,
                    "parent_chunk_id": l1_id,
                    "root_chunk_id": l1_id,
                    "chunk_level": 2,
                    **metadata
                })

                # Layer 3: Small Chunks (Chunk3) - Split from L2
                layer3_chunks = self.l3_splitter.split_text(l2_text)
                for l3_idx, l3_text in enumerate(layer3_chunks):
                    l3_id = f"{l2_id}_l3_{l3_idx}"
                    results.append({
                        "text": l3_text,
                        "chunk_id": l3_id,
                        "parent_chunk_id": l2_id,
                        "root_chunk_id": l1_id,
                        "chunk_level": 3,
                        **metadata
                    })

        return results

    def _simple_split(self, text: str, size: int, overlap: int) -> List[str]:
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
            if start >= len(text):
                break
        return chunks

    def deduplicate_layer2(self, chunks: List[Dict]) -> List[Dict]:
        """
        Placeholder for Layer 2 deduplication logic as requested by user.
        Currently uses simple exact text hashing for deduplication.
        """
        import hashlib
        seen_hashes = set()
        unique_chunks = []
        for chunk in chunks:
            text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
            if text_hash not in seen_hashes:
                unique_chunks.append(chunk)
                seen_hashes.add(text_hash)
        return unique_chunks
