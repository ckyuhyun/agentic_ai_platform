import operator

from typing import Annotated, Optional, List, Union
from pydantic import BaseModel, Field, model_validator
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from agentic_ai_platform.states.hallucination_signal_state import HallucinationCheckerConfig
from agentic_ai_platform.states.plan_state import PlanState
from agentic_ai_platform.states.queryState import QueryState
from agentic_ai_platform.states.tool_state import ToolState
from agentic_ai_platform.states.filter_message_state import FilterMessageItem
from agentic_ai_platform.graph.node_trace import NodeTrace



class CriticFeedback(BaseModel):
    """Structured output from the critic node."""
    score: Annotated[float, operator.add] = Field(ge=0.0, le=1.0, description="Quality score from 0 (poor) to 1 (excellent)")
    approved: Annotated[bool, operator.add] = Field(description="Whether the draft meets the acceptance threshold")
    issues: Annotated[List[str], operator.add] = Field(default_factory=list, description="Specific problems found in the draft")
    suggestions: Annotated[List[str], operator.add] = Field(default_factory=list, description="Concrete improvements to apply")
    reasoning: Annotated[str, operator.add] = Field(description="Brief explanation of the score and decision")

    hallucination_config : Annotated[HallucinationCheckerConfig, Field(default_factory=HallucinationCheckerConfig, description="Configuration for hallucination checking")]
    hallucination_score : Annotated[Optional[float], Field(default=None, description="Hallucination severity score from 0 (none) to 1 (severe)")]
    hallucination_issues: Annotated[Optional[List[str]], Field(default=None, description="List of identified hallucination issues, if any")]

class DraftConfig(BaseModel):
    """Configuration parameters for the drafting process. and allowing writign once"""
    max_iterations: Annotated[int, Field(default=3, description="Maximum number of draft/critique cycles before forcing acceptance")]
    approval_threshold: Annotated[float, Field(default=0.8, description="Minimum critic score required for approval")]

    # _initialized: bool = False 
    # @model_validator(mode="after")
    # def check_immutable_fields(self):
    #     if self._initialized:
    #         raise ValueError("DraftConfig is immutable and cannot be modified after initialization")
    #     self._initialized = True
    #     return self

    
class AbstractSuperviseState(BaseModel):
    # Observability/eval identifiers, set once when the run starts and carried
    # unchanged through every node so LLM calls, tool calls, and stored records
    # can be grouped and replayed by run.
    thread_id : Annotated[str, Field(
         default=None, description="Thread Id of this graph runthread id")]

    # state_id: Optional[str] = Field(
    #     default=None, description="Id of this graph run (== LangGraph/scheduler thread id)")

    # session_id: Optional[str] = Field(
    #     default=None, description="Business-level grouping id (defaults to state_id; distinct when multiple runs belong to one session)")


    iteration: Annotated[int, Field(
        default=0, description="Number of draft/critique cycles completed")]

    # LangGraph message history
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    # Trace-based evaluation records
    node_traces: Annotated[List[NodeTrace], Field(
        default_factory=list, description="Per-node execution traces for evaluation")]

    messages_filtered : Annotated[bool, Field(
            default=False, description="if the slack messages filtered out to remove unnecessary messages, return True")]

    filtered_message : Annotated[List[FilterMessageItem], Field(default_factory = list,  description="filtered message with different aspect - blacklist, hallucination, safty etc")]



class SuperviseState(AbstractSuperviseState):
    """State shared between the drafter and critic nodes."""

    
    
    query_state : Annotated[QueryState, Field(
        default_factory=QueryState, description="State related to query rewriting and generation")]

    # Drafter output
    draft: Annotated[Optional[Union[str,list]], Field(
        default=None, description="Most recent draft produced by the drafter")]    
    

    # Critic output
    critique: Annotated[Optional[CriticFeedback], Field(
        default=None, description="Structured feedback from the critic")]

    # Tool output history (if using tools)
    tool_calls: Annotated[list[ToolState], Field(
        default_factory=list, description="History of tool calls made during drafting, if any")]

    # Loop control
    graph_config : Annotated[DraftConfig, Field(
        default_factory=DraftConfig, description="Configuration for the drafting process")]

    

    # Final result — set when critic approves or max_iterations reached
    final_output: Annotated[Optional[str], Field(
        default=None, description="The accepted draft")]

    


    # Cursor into `messages` marking how many have been consumed by downstream
    # agents (e.g. rewrite_query_agent picking up new human_review answers).
    last_reviewed_message_index: Annotated[int, Field(
        default=0, description="Number of messages already processed from the message history")]

    # planner state
    plan : Annotated[PlanState, Field(
        default_factory=PlanState, description="planning for execution")]


