from pydantic import BaseModel, Field

from agent.persona import experence

class PersonaState(BaseModel):
    core: dict=Field(..., description="核心人格信息，包括基本信息、兴趣爱好、性格特点等")
    experience: dict=Field(..., description="经历信息，包括重要的生活经历、成长背景等")
    relationship: dict=Field(..., description="人际关系信息，包括与他人的互动历史、情感联系等")
    memories: list=Field(..., description="记忆信息，包括过往经历、重要事件等")