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

    def create_collection(self, dense_dim: int = 1024):
        client = self._get_connect()
        if client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text_dense", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text", DataType.VARCHAR, max_length=16384, enable_analyzer=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=512)
        schema.add_field("filename", DataType.VARCHAR, max_length=255)
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_level", DataType.INT64)
        schema.add_field("title_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("content_type", DataType.VARCHAR, max_length=64)
        schema.add_field("page_number", DataType.INT64)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["text_sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="text_dense",
            index_name="text_dense_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="text_sparse",
            index_name="text_sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger_handler.info(f"Collection '{self.collection_name}' created successfully with schema and indexes.")

    def insert(self, data: list[dict]):
        return self._get_connect().insert(self.collection_name, data)

    def delete(self, filter_expr: str):
        return self._get_connect().delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def query_by_chunk_ids(self, chunk_ids: list[str], output_fields: list[str] | None = None) -> list[dict]:
        if not chunk_ids:
            return []
        client = self._get_connect()
        normalized_ids = [chunk_id.replace('"', '\\"') for chunk_id in chunk_ids]
        quoted_ids = ", ".join(f'"{chunk_id}"' for chunk_id in normalized_ids)
        filter_expr = f"chunk_id in [{quoted_ids}]"
        return client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["chunk_id"],
        )

    def has_chunk(self, chunk_id: str) -> bool:
        results = self.query_by_chunk_ids([chunk_id], output_fields=["chunk_id"])
        return len(results) > 0

    def has_collection(self) -> bool:
        return self._get_connect().has_collection(self.collection_name)

    def drop_collection(self):
        client = self._get_connect()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)


if __name__ == "__main__":
    milvus_manager = MilvusManage()
    milvus_manager.create_collection()
    milvus_manager.disconnect()
