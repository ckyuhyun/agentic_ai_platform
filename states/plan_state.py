
from typing import List, Dict, Any
from pydantic import BaseModel, Field



class PlanState(BaseModel):
    input:str = Field(default="", description="Original user query or rewritten query to be planned for execution") 
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="Current list of steps")
    #reasoning: str = Field(description="")
    past_steps: List[str] = Field(default_factory=list, description="Results from completed tasks") 


