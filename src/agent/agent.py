from agent.graph import build_graph
from agent.utils.logger_handler import get_logger

logger = get_logger()


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
