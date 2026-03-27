from typing import Annotated, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class CriticFeedback(BaseModel):
    """Structured output from the critic node."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score from 0 (poor) to 1 (excellent)")
    approved: bool = Field(description="Whether the draft meets the acceptance threshold")
    issues: List[str] = Field(default_factory=list, description="Specific problems found in the draft")
    suggestions: List[str] = Field(default_factory=list, description="Concrete improvements to apply")
    reasoning: str = Field(description="Brief explanation of the score and decision")


class DraftState(BaseModel):
    """State shared between the drafter and critic nodes."""

    # Task definition
    task: str = Field(description="The task or prompt the drafter must complete")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-level instruction to shape both drafter and critic behaviour"
    )

    # Drafter output
    draft: Optional[str] = Field(default=None, description="Most recent draft produced by the drafter")

    # Critic output
    critique: Optional[CriticFeedback] = Field(default=None, description="Structured feedback from the critic")

    # Loop control
    iteration: int = Field(default=0, description="Number of draft/critique cycles completed")
    max_iterations: int = Field(default=3, description="Maximum allowed draft/critique cycles before forcing acceptance")
    approval_threshold: float = Field(default=0.8, description="Minimum critic score required for approval")

    # Final result — set when critic approves or max_iterations reached
    final_output: Optional[str] = Field(default=None, description="The accepted draft")

    # LangGraph message history
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
