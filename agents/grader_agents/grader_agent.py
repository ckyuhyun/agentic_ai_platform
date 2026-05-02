from typing import Type, List, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from agentic_ai_platform.state_manager.draft_state import DraftState, NodeTrace
from agentic_ai_platform.tools.grader_tools import EvalsTools


DEFAULT_GRADER_TOOLS: List[BaseTool] = [
    EvalsTools.check_constraints,
    EvalsTools.check_hallucinations,
    EvalsTools.check_efficiency,
    EvalsTools.check_ethical_considerations,
]

# Maps a tool name to the input key it expects
_TOOL_INPUT: dict = {
    "check_constraints":           "content",
    "check_hallucinations":        "draft",
    "check_efficiency":            "draft",
    "check_ethical_considerations": "draft",
}







def create_grader_agent(
    schema: Type[BaseModel],
    system_prompt: list,
    graph_llm=None,
    eval_tools: Optional[List[BaseTool]] = None,
):
    """
    Factory that returns a grader node bound to the given Pydantic schema.

    schema must have: score (float), approved (bool), issues (list[str]).

    eval_tools: list of grader tools to run before the LLM call. Each tool's
                report is appended to the prompt so the LLM can use the findings.
                Defaults to all four GraderTools checks.
                Pass an empty list [] to skip tool checks entirely.
    """
    #active_tools: List[BaseTool] = DEFAULT_GRADER_TOOLS if tools is None else tools
    #tool_map: dict[str, BaseTool] = {t.name: t for t in eval_tools} if eval_tools is not None else {}

    #structured_model = graph_llm.with_structured_output(schema)

    def grader_node(state: DraftState):

        # start updating trace
        trace = NodeTrace.start(node="grader", iteration=state.iteration, model="llama3.1")

        #tool_reports = _run_eval_tools(tool_map, state)        
        #tool_reports = run_eval_tools()

        # prompt = system_prompt + [
        #     HumanMessage(content=_build_eval_message(state, tool_reports)),
        # ]
        prompt = system_prompt + [HumanMessage(content=f"Task:\n{state.task}\n\nDraft:\n{state.draft}")]

        critic_llm = graph_llm.bind_tools(eval_tools)
        try:
            feedback = critic_llm.invoke(prompt)
        
            # feedback.approved = (
            #     feedback.score >= state.drafter_config.approval_threshold
            #     and feedback.issues == []
            # )
        except Exception as e:
            RuntimeError(f"Error invoking grader LLM: {e}")

        # final = None
        # if feedback.approved or state.iteration >= state.drafter_config.max_iterations:
        #     final = state.draft

        # feedback.hallucination_score = 0.1
        # state.critique = feedback
        # state.final_output = final
        
        # state.messages = []
        # state.node_traces.append(
        #     trace.finish(
        #         score=feedback.score,
        #         approved=feedback.approved,
        #         issue_count=len(feedback.issues),
        #     )
        # )
        
        state.critique = feedback
        return {'messages' : [feedback]}
    
    return grader_node
    

    
def run_eval_tools():
    pass 


def _run_eval_tools(tool_map: dict, state: DraftState) -> dict[str, str]:
    """Invoke every tool in tool_map and return {tool_name: report}."""
    reports = {}
    try:
        for name, tool in tool_map.items():
            tool_input = _build_tool_input(tool, state)
            if name == "check_hallucinations":
                tool_input["state"] = state
            reports[name] = tool.invoke(tool_input)
    except Exception as e:
        raise RuntimeError(f"Error invoking tool '{name}': {e}")

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
