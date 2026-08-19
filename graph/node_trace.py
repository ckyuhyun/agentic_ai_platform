from typing import Annotated, Optional, List, Union
from pydantic import BaseModel, Field
import time

class NodeTrace(BaseModel):
    """Execution record written by each node for trace-based evaluation."""
    node: Annotated[str, Field(description="Node name")]
    iteration: Annotated[int, Field(description="recycle index of node calling")]
    started_at: Annotated[float, Field(default_factory=time.time)]
    latency_ms: Annotated[float, Field(default=0.0)]
    model: Annotated[str, Field(default="")]
    tool_calls_made: Annotated[List[str], Field(default_factory=list)]
    draft_len: Annotated[Optional[int], Field(default=None)]
    score: Annotated[Optional[float], Field(default=None)]
    approved: Annotated[Optional[bool], Field(default=None)]
    issue_count: Annotated[Optional[int], Field(default=None)]
 
    @staticmethod
    def start(node: str, 
              iteration: int, 
              model: str = "") -> "NodeTrace":
        return NodeTrace(node=node, iteration=iteration, model=model)    
    

    def finish(self, 
               **kwargs) -> "NodeTrace":
        self.latency_ms = (time.time() - self.started_at) * 1000
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self
