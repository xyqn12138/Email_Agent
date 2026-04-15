from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

chat_model = ChatTongyi(
    model="qwen-plus"
)

embedding_model = DashScopeEmbeddings(
    model="text-embedding-v4"
)