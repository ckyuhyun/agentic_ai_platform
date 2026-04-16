import operator
from typing import Annotated, Optional, List, Union
from pydantic import BaseModel, Field, model_validator
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from agentic_ai_platform.state_manager.tool_state import ToolState



class CriticFeedback(BaseModel):
    """Structured output from the critic node."""
    score: Annotated[float, operator.add] = Field(ge=0.0, le=1.0, description="Quality score from 0 (poor) to 1 (excellent)")
    approved: Annotated[bool, operator.add] = Field(description="Whether the draft meets the acceptance threshold")
    issues: Annotated[List[str], operator.add] = Field(default_factory=list, description="Specific problems found in the draft")
    suggestions: Annotated[List[str], operator.add] = Field(default_factory=list, description="Concrete improvements to apply")
    reasoning: Annotated[str, operator.add] = Field(description="Brief explanation of the score and decision")
    

class DraftConfig(BaseModel):
    """Configuration parameters for the drafting process. and allowing writign once"""
    max_iterations: int = Field(default=3, description="Maximum number of draft/critique cycles before forcing acceptance")
    approval_threshold: float = Field(default=0.8, description="Minimum critic score required for approval")

    # _initialized: bool = False 
    # @model_validator(mode="after")
    # def check_immutable_fields(self):
    #     if self._initialized:
    #         raise ValueError("DraftConfig is immutable and cannot be modified after initialization")
    #     self._initialized = True
    #     return self

    
    


class DraftState(BaseModel):
    """State shared between the drafter and critic nodes."""

    # Task definition
    task: Annotated[str, operator.add] = Field(description="The task or prompt the drafter must complete")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-level instruction to shape both drafter and critic behaviour"
    )

    # Drafter output
    draft: Optional[Union[str,list]] = Field(default=None, description="Most recent draft produced by the drafter")

    # Critic output
    critique: Optional[CriticFeedback] = Field(default=None, description="Structured feedback from the critic")

    # Tool output history (if using tools)
    tool_calls: list[ToolState] = Field(default_factory=list, description="History of tool calls made during drafting, if any")

    # Loop control
    drafter_config : DraftConfig = Field(default_factory=DraftConfig, description="Configuration for the drafting process")
    iteration: int = Field(default=0, description="Number of draft/critique cycles completed")
    

    # Final result — set when critic approves or max_iterations reached
    final_output: Optional[str] = Field(default=None, description="The accepted draft")

    # LangGraph message history
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # LangSmith tracing metadata
    node_traces : List[dict] = Field(default_factory=list, description="Execution traces for LangSmith monitoring and debugging")
