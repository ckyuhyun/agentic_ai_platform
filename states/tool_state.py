from typing import Annotated, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

class ToolState(BaseModel):
    query : Annotated[str, Field(description="The query or input provided to the tool")]
    tool_name : Annotated[str, Field(description="Name of the tool")]
    tool_args : Annotated[Dict[str, Any], Field(description="Arguments for the tool")]
    tool_result : Annotated[Any, Field(description="Result returned by the tool", default="")]
    status : Annotated[Literal["success", "failed"],Field(default="failed", description="Outcome of this tool call attempt") ]
    attempt : Annotated[int, Field(default=1, description="1-indexed attempt number for this tool_name")]
    error: Annotated[Optional[str], Field(default=None,  description="Error message if status is 'failed'")]




