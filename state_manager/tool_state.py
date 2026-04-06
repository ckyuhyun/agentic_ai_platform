from typing import Dict, Any
from pydantic import BaseModel, Field

class ToolState(BaseModel):
    query : str = Field(description="The query or input provided to the tool")
    tool_name : str = Field(description="Name of the tool")
    tool_args : Dict[str, Any] = Field(description="Arguments for the tool")
    tool_result : Any = Field(description="Result returned by the tool", default="")




