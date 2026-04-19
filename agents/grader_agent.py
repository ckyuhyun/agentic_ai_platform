from typing import Type
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_platform.state_manager.draft_state import DraftState, NodeTrace


_CRITIC_SYSTEM = (
    "You are a strict but fair critic. Evaluate the draft against the original task. "
    "Return your evaluation as structured data matching the requested schema exactly."
)


def _route(state: DraftState) -> str:
    if state.final_output is not None:
        return "end"
    if state.critique and state.critique.approved:
        return "end"
    if state.iteration >= state.drafter_config.max_iterations:
        return "human_review"
    return "drafter"


def create_grader_agent(schema: Type[BaseModel], graph_llm=None):
    """
    Factory that returns a grader node bound to the given Pydantic schema.

    schema must have: score (float), approved (bool), issues (list[str]).
    """
    structured_model = graph_llm.with_structured_output(schema)

    def grader_node(state: DraftState) -> DraftState:
        trace = NodeTrace.start(node="grader", iteration=state.iteration, model="llama3.1")

        prompt = [
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=(
                f"Original task:\n{state.task}\n\n"
                f"Draft to evaluate:\n{state.draft}\n\n"
                f"Approval threshold: {state.drafter_config.approval_threshold}"
            )),
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

    return grader_node
