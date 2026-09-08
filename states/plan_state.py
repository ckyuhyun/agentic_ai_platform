
from typing import Annotated, List, Dict, Any
from pydantic import BaseModel, Field



class PlanState(BaseModel):
    input: Annotated[str, Field(default="", description="Original user query or rewritten query to be planned for execution")]
    plan: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Current list of steps")]
    non_plan_steps: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Steps that were considered but not included in the plan, with reasoning for why each was rejected")]
    #reasoning: str = Field(description="")
    past_steps: Annotated[List[str], Field(default_factory=list, description="Results from completed tasks")]


