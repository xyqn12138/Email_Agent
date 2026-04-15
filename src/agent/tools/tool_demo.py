from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from zai import ZhipuAiClient
import asyncio
from agent.schema.tools_schema import web_search_schema as WebSearchInput

load_dotenv()  # 加载环境变量


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "一个网络搜索工具函数,你可以使用它来搜索互联网上的信息。"

    args_schema : Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            client = ZhipuAiClient(
                api_key=os.getenv("ZHIPU_API_KEY")
            )
            response = client.web_search.web_search(
                search_engine="search_pro",
                search_query=query,  
                search_recency_filter="noLimit",
            )
            if response.search_result:
                return "\n\n".join([item.content for item in response.search_result])
            else:
                return "No results found."
        except Exception as e:
            return f"Error occurred while searching: {e}"
    

    async def _arun(self, query: str) -> str:
        return await asyncio.to_thread(self._run, query)