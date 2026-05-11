from agent.graph import build_graph
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


def run_chat():
    agent = build_graph()
    print("=== 个人超级知识库 Agent ===")
    print("输入问题开始对话，输入 'quit' 退出\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            ai_message = result["messages"][-1]
            print(f"\nAgent: {ai_message.content}\n")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            print(f"\n[错误] {e}\n")


if __name__ == "__main__":
    run_chat()
