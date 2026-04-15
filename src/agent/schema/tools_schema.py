from pydantic import BaseModel, Field

class web_search_schema(BaseModel):
    query: str = Field(..., description="搜索查询字符串")
    