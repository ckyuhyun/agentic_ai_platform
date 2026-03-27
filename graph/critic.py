from typing import Type
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_platform.model.llm import llm
from agentic_ai_platform.state_manager.draft_state import DraftState, CriticFeedback
from agentic_ai_platform.utils.color_print import cprint, C


_CRITIC_SYSTEM = (
    "You are a strict but fair critic. Evaluate the draft against the original task. "
    "Return your evaluation as structured data matching the requested schema exactly."
)

def _route(state: DraftState) -> str:
    """
    Routing function for the conditional edge after the critic node.

    Returns:
        "end"     — critic approved or max iterations reached
        "drafter" — critic rejected and iterations remain
    """
    if state.final_output is not None:
        return "end"
    if state.critique and state.critique.approved:
        return "end"
    if state.iteration >= state.max_iterations:
        return "end"
    return "drafter"


def make_critic_node(schema: Type[BaseModel]):
    """
    Factory that returns a critic node bound to the given Pydantic schema.

    The returned node calls with_structured_output(schema) so the LLM is
    forced to produce a valid instance of that model — no JSON parsing needed.

    Args:
        schema: Any Pydantic BaseModel class. Must have an `approved` bool field
                and a `score` float field for the routing logic to work.

    Usage:
        # Default — uses CriticFeedback
        critic_node = make_critic_node()

        # Custom schema
        class MyFeedback(BaseModel):
            score: float
            approved: bool
            notes: str

        critic_node = make_critic_node(MyFeedback)
    """
    def critic_node(state: DraftState) -> dict:
        base_model = llm("llama3.1").llm_instance
        structured_model = base_model.with_structured_output(schema)
        

        prompt = [
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=(
                f"Original task:\n{state.task}\n\n"
                f"Draft to evaluate:\n{state.draft}\n\n"
                f"Approval threshold: {state.approval_threshold}"
            )),
        ]

        feedback = structured_model.invoke(prompt)
        cprint(f"\n── Critic Feedback ──────────────────────────────")
        cprint(f"=> Score: {feedback.score:.2f}", C.CYAN)
        cprint(f"=> Approved: {feedback.approved}", C.CYAN)
        cprint(f"=> Issues:\n" + "\n".join(f"   - {i}" for i in feedback.issues), C.CYAN)
        cprint(f"=> Suggestions:\n" + "\n".join(f"   - {s}" for s in feedback.suggestions), C.CYAN)
        cprint(f"=> Reasoning:\n   {feedback.reasoning}", C.CYAN)
        print(f"\n──────────────────────────────────────────────────")

        # Enforce threshold against the score field
        feedback.approved = feedback.score >= state.approval_threshold


        final = None
        if feedback.approved or state.iteration >= state.max_iterations:
            final = state.draft

        return {
            "critique": feedback,
            "final_output": final,
            "messages": [],
        }

    return critic_node



