
from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field



class PlanState(BaseModel):
    input: Annotated[str, Field(default="", description="Original user query or rewritten query to be planned for execution")]
    plan: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Current list of steps")]
    non_plan_steps: Annotated[List[Dict[str, Any]], Field(default_factory=list, description="Steps that were considered but not included in the plan, with reasoning for why each was rejected")]
    #reasoning: str = Field(description="")
    past_steps: Annotated[List[str], Field(default_factory=list, description="Results from completed tasks")]

    next_step_override: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="Ad-hoc step (same shape as an entry in `plan`) that a node can set on the state it "
                    "returns to redirect execution based on what it learned at runtime -- e.g. a step's "
                    "own result implying a different agent should run next than the static plan says. "
                    "A dispatcher that supports it should splice this in as the next step to run and "
                    "clear the field; dispatchers that don't consume it can safely ignore it.")]

    on_failure_override: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="Ad-hoc step (same shape as an entry in `plan`) that acts as an armed safety net: "
                    "any node can set this on the state it returns, and it stays armed across later "
                    "steps until a dispatcher consumes it. A dispatcher that supports failover should "
                    "check it whenever a step fails (raises, or exhausts its retry budget) and, if set, "
                    "route to it instead of escalating to a generic human-review/interrupt path -- then "
                    "clear the field, since it's a one-shot redirect, not a standing rule.")]

    step_error: Annotated[Optional[str], Field(
        default=None,
        description="Set by the dispatcher when a step's agent call raises an uncaught exception (as "
                    "opposed to a tool reporting a normal failed ToolState), so routing/human-review "
                    "logic that otherwise only inspects tool_states has a signal to react to. Cleared "
                    "once a human-review decision (or an on_failure_override reroute) resolves it.")]


