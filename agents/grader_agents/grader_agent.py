from typing import Type, List, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from agentic_ai_platform.state_manager.draft_state import DraftState, NodeTrace
from agentic_ai_platform.tools.grader_tools import GraderTools


DEFAULT_GRADER_TOOLS: List[BaseTool] = [
    GraderTools.check_constraints,
    GraderTools.check_hallucinations,
    GraderTools.check_efficiency,
    GraderTools.check_ethical_considerations,
]

# Maps a tool name to the input key it expects
_TOOL_INPUT: dict = {
    "check_constraints":           "content",
    "check_hallucinations":        "draft",
    "check_efficiency":            "draft",
    "check_ethical_considerations": "draft",
}




def _route(state: DraftState) -> str:
    if state.final_output is not None:
        return "end"
    if state.critique and state.critique.approved:
        return "end"
    if state.iteration >= state.drafter_config.max_iterations:
        return "human_review"
    return "drafter"


def create_grader_agent(
    schema: Type[BaseModel],
    system_prompt: list,
    graph_llm=None,
    tools: Optional[List[BaseTool]] = None,
):
    """
    Factory that returns a grader node bound to the given Pydantic schema.

    schema must have: score (float), approved (bool), issues (list[str]).

    tools: list of grader tools to run before the LLM call. Each tool's
           report is appended to the prompt so the LLM can use the findings.
           Defaults to all four GraderTools checks.
           Pass an empty list [] to skip tool checks entirely.
    """
    #active_tools: List[BaseTool] = DEFAULT_GRADER_TOOLS if tools is None else tools
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools} if tools is not None else {}

    structured_model = graph_llm.with_structured_output(schema)

    def grader_node(state: DraftState) -> DraftState:
        trace = NodeTrace.start(node="grader", iteration=state.iteration, model="llama3.1")

        tool_reports = _run_tools(tool_map, state)        

        prompt = system_prompt + [
            HumanMessage(content=_build_eval_message(state, tool_reports)),
        ]

        feedback = structured_model.invoke(prompt)
        feedback.approved = (
            feedback.score >= state.drafter_config.approval_threshold
            and feedback.issues == []
        )

        final = None
        if feedback.approved or state.iteration >= state.drafter_config.max_iterations:
            final = state.draft

        state.critique = feedback
        state.final_output = final
        state.messages = []
        state.node_traces.append(
            trace.finish(
                score=feedback.score,
                approved=feedback.approved,
                issue_count=len(feedback.issues),
            )
        )
        return state
    

    



def _run_tools(tool_map: dict, state: DraftState) -> dict[str, str]:
    """Invoke every tool in tool_map and return {tool_name: report}."""
    reports = {}
    for name, tool in tool_map.items():
        tool_input = _build_tool_input(tool, state)
        reports[name] = tool.invoke(tool_input)
    return reports


def _build_tool_input(tool: BaseTool, state: DraftState) -> dict:
    """Return the correct input dict for a given grader tool."""
    key = _TOOL_INPUT.get(tool.name, "draft")
    if key == "content":
        return {"content": f"TASK: {state.task}\nDRAFT: {state.draft}"}
    return {"draft": state.draft or ""}


def _build_eval_message(state: DraftState, tool_reports: dict[str, str]) -> str:
    base = (
        f"Original task:\n{state.task}\n\n"
        f"Draft to evaluate:\n{state.draft}\n\n"
        f"Approval threshold: {state.drafter_config.approval_threshold}"
    )
    if not tool_reports:
        return base

    sections = "\n\n".join(
        f"--- {name} ---\n{report}" for name, report in tool_reports.items()
    )
    return f"{base}\n\n{sections}"
