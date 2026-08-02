from typing import Annotated, Dict, Any
from pydantic import BaseModel, Field

class ToolState(BaseModel):
    query : Annotated[str, Field(description="The query or input provided to the tool")]
    tool_name : Annotated[str, Field(description="Name of the tool")]
    tool_args : Annotated[Dict[str, Any], Field(description="Arguments for the tool")]
    tool_result : Annotated[Any, Field(description="Result returned by the tool", default="")]




