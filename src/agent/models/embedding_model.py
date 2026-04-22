import requests
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    def __init__(self, model_name: str = "zhipuai", dimensions: int = 1024):
        self.model_name = model_name
        self.dimensions = dimensions

    def get_zhipu_embedding(self, text: str, model: str = "embedding-3", dimensions: int = 1024):
        url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        headers = {
            "Authorization": f"Bearer {os.getenv('ZHIPUAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": text,
            "dimensions": dimensions   # embedding-3 支持指定维度
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def get_dashscope_embedding(self, text: str, model: str = "text-embedding-v4", dimensions: int = 1024):
        uri = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
            "Content-Type": "application/json"  
        }
        payload = { 
            "model": model,
            "input": text,
            "dimensions": dimensions
        }

        response = requests.post(uri, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_query(self, text: str):
        if self.model_name == "zhipuai":
            return self.get_zhipu_embedding(text, dimensions=self.dimensions)
        elif self.model_name == "dashscope":
            return self.get_dashscope_embedding(text, dimensions=self.dimensions)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def embed_documents(self, texts: list):
        return [self.embed_query(text) for text in texts]

    

if __name__ == "__main__":
    txt = "这是一个测试文本。"
    embedding_model = EmbeddingModel(model_name="dashscope", dimensions=1024)
    embedding = embedding_model.embed_query(txt)
    print(f"Embedding for '{txt}': {embedding[:10]}...") 