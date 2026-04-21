from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
import os 
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatTongyi(
    model="qwen-plus"
)

def get_embedding_model(name: str):
    if name == "dashscope":
        return DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    elif name == "zhipuai":
        return ZhipuAIEmbeddings(
            model="embedding-3",
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            dimensions=1024
        )
    else:
        raise ValueError(f"Unknown embedding model: {name}")
