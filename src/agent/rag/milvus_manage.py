from pymilvus import MilvusClient, DataType, Function, FunctionType
from dotenv import load_dotenv
from agent.utils.logger_handler import get_logger
import os
load_dotenv()
logger_handler = get_logger()
class MilvusManage:
    def __init__(self):
        self.host = os.getenv('MILVUS_HOST')
        self.port = int(os.getenv('MILVUS_PORT'))
        self.uri = os.getenv('MILVUS_URI')
        self.collection_name = os.getenv('COLLECTION_NAME')
        self.client = None
    
    def _get_connect(self) -> MilvusClient:
        if self.client is None:
            self.client = MilvusClient(uri=self.uri)
            print("Connected to Milvus successfully.")
        return self.client
    
    def disconnect(self):
        if self.client is not None:
            self.client.close()
            self.client = None
            print("Disconnected from Milvus.")
    
    
    def create_collection(self, dense_dim: int=1024):
        client = self._get_connect()
        if client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        # 主键
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        
        # 密集向量（来自 embedding 模型）
        schema.add_field("text_dense", DataType.FLOAT_VECTOR, dim=dense_dim)
        
        # 稀疏向量（来自 BM25）
        schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        
        # 文本和元数据字段
        schema.add_field("text", DataType.VARCHAR, max_length=5120, enable_analyzer=True)
        schema.add_field("filename", DataType.VARCHAR, max_length=255)
        schema.add_field("file_type", DataType.VARCHAR, max_length=50)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)

        # Auto-merging 所需层级字段
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_level", DataType.INT64)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["text_sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # 为两种向量分别创建索引
        index_params = client.prepare_index_params()
        
        # 密集向量索引 - 使用 HNSW（更适合混合检索）
        index_params.add_index(
            field_name="text_dense",
            index_name="text_dense_index",
            index_type="AUTOINDEX",
            metric_type="IP",
            # params={"M": 16, "efConstruction": 256}
        )
        
        # 稀疏向量索引
        index_params.add_index(
            field_name="text_sparse",
            index_name="text_sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger_handler.info(f"Collection '{self.collection_name}' created successfully with schema and indexes.")
        

    def insert(self, data: list[dict]):
        """插入数据到 Milvus"""
        return self._get_connect().insert(self.collection_name, data)

    def delete(self, filter_expr: str):
        """删除数据"""
        return self._get_connect().delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self._get_connect().has_collection(self.collection_name)

    def drop_collection(self):
        """删除集合（用于重建 schema）"""
        client = self._get_connect()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)


if __name__ == "__main__":
    milvus_manager = MilvusManage()
    milvus_manager.create_collection()
    milvus_manager.disconnect()