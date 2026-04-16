from typing import Type
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_ai_platform.llm.llm import LLM
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
    # if state.iteration >= state.drafter_config.max_iterations:
    #     return "human_review"
    return "drafter"


def create_grader_agent(schema: Type[BaseModel],
                        tool_llm=None,
                        graph_llm=None,
                        tools=None):
    """
    Factory that returns a grader node bound to the given Pydantic schema.

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
    def critic_node(state: DraftState) -> DraftState:
        
        structured_model = graph_llm.with_structured_output(schema)
        

        prompt = [
            SystemMessage(content=_CRITIC_SYSTEM),
            HumanMessage(content=(
                f"Original task:\n{state.task}\n\n"
                f"Draft to evaluate:\n{state.draft}\n\n"
                f"Approval threshold: {state.drafter_config.approval_threshold}"
            )),
        ]

        feedback = structured_model.invoke(prompt)
        #cprint(f"\n── Critic Feedback ──────────────────────────────")
        #cprint(f"=> Score: {feedback.score:.2f}", C.CYAN)
        #cprint(f"=> Approved: {feedback.approved}", C.CYAN)
        #cprint(f"=> Issues:\n" + "\n".join(f"   - {i}" for i in feedback.issues), C.CYAN)
        #cprint(f"=> Suggestions:\n" + "\n".join(f"   - {s}" for s in feedback.suggestions), C.CYAN)
        #cprint(f"=> Reasoning:\n   {feedback.reasoning}", C.CYAN)
        #print(f"\n──────────────────────────────────────────────────")

        # Enforce threshold against the score field
        feedback.approved = feedback.score >= state.drafter_config.approval_threshold and feedback.issues == []


        final = None
        if feedback.approved or state.iteration >= state.drafter_config.max_iterations:
            final = state.draft

        state.critique = feedback
        state.final_output = final
        state.messages = []

        return state

    return critic_node



