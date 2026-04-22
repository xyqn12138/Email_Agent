from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatTongyi(
    model="qwen-plus"
)