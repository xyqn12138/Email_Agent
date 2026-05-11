from langchain.agents import create_agent
from agent.models.chat_model import create_chat_model
from agent.tools.web_tool import web_search
from agent.tools.rag_tool import knowledge_base_search
from agent.utils.logger_handler import get_logger

logger = get_logger()

SYSTEM_PROMPT = (
    "你是一个个人超级知识库助手。你拥有以下工具：\n"
    "1. knowledge_base_search：从个人知识库中检索教材、课件等学习资料。"
    "当用户的问题涉及课程内容、教材知识点时，优先使用此工具。\n"
    "2. web_search：搜索互联网获取最新信息。\n\n"
    "使用规则：\n"
    "- 优先从知识库检索，知识库没有相关内容时再使用网络搜索\n"
    "- 回答要基于检索到的资料，不要编造\n"
    "- 如果资料不足以回答，如实告知用户\n"
    "- 回答使用中文，条理清晰"
)


def build_graph(provider: str | None = None, model: str | None = None):
    llm = create_chat_model(provider=provider, model=model)
    tools = [knowledge_base_search, web_search]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


graph = build_graph()
