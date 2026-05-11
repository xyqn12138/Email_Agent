from .rag.retriever import Retriever
from .tools.web_tool import web_search
from .utils.logger_handler import get_logger

logger = get_logger()


def build_agent(provider: str | None = None, model: str | None = None):
    from .models.chat_model import create_chat_model
    llm = create_chat_model(provider=provider, model=model)
    tools = [web_search]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools


def ask_with_rag(question: str, provider: str | None = None, top_k: int = 3):
    from .models.chat_model import create_chat_model
    retriever = Retriever()
    contexts = retriever.retrieve(question, limit=top_k)

    context_block = ""
    for i, ctx in enumerate(contexts, 1):
        context_block += (
            f"\n---\n[{i}] {ctx['title_path']} (page {ctx['page_number']})\n"
            f"{ctx['chunk3_text'] or ctx['search_hit']}\n"
        )

    prompt = (
        "你是一个学习助手。根据以下参考资料回答用户问题。"
        "如果参考资料不足以回答，请说明。\n\n"
        f"参考资料：{context_block}\n\n"
        f"用户问题：{question}"
    )

    llm = create_chat_model(provider=provider)
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    question = "快速排序的时间复杂度是多少？"
    print(f"=== Retrieval Only (no LLM) ===")
    print(f"Query: {question}\n")

    retriever = Retriever()
    contexts = retriever.retrieve(question, limit=5, skip_rewrite=True)

    for i, ctx in enumerate(contexts, 1):
        print(f"--- [{i}] {ctx['title_path']} (page {ctx['page_number']}) ---")
        print(f"  Level:   {ctx['level']}")
        print(f"  File:    {ctx['filename']}")
        if ctx['chunk1_text']:
            print(f"  L1 (章): {ctx['chunk1_text'][:120]}...")
        if ctx['chunk2_text']:
            print(f"  L2 (节): {ctx['chunk2_text'][:120]}...")
        hit_text = ctx['chunk3_text'] or ctx['search_hit']
        print(f"  Hit:     {hit_text[:200]}...")
        print()
