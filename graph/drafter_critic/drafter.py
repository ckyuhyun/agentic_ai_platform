from typing import Type
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_ai_platform.model.llm import llm
from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.utils.color_print import C, cprint


_DEFAULT_SYSTEM = (
    "You are a skilled writer. Complete the task given to you as thoroughly and clearly as possible. "
    "If you receive feedback from a previous critique, revise your draft to address every issue raised."
)

def make_drafter_node(schema : Type[BaseModel]):
    """
    Factory that returns a drafter node bound to the given Pydantic schema.

    The returned node produces drafts for the given task, incorporating any critic feedback on subsequent iterations.

    Args:
        schema: The Pydantic model class representing the state. Must include `task`, `draft`, `iteration`, and `critique` fields.
                Default is DraftState, but you can use a custom model as long as it has those fields.
    """
    def drafter_node(state: DraftState) -> DraftState:
        model = llm(state.system_prompt or "llama3.1").llm_instance

        messages = []

        system_text = state.system_prompt or _DEFAULT_SYSTEM
        messages.append(SystemMessage(content=system_text))

        if state.iteration == 0 or state.critique is None:
            # First draft
            messages.append(HumanMessage(content=f"Task:\n{state.task}"))
        else:
            # Revision — include previous draft and critique
            critique = state.critique
            revision_prompt = (
                f"Task:\n{state.task}\n\n"
                f"Your previous draft:\n{state.draft}\n\n"
                f"Critic score: {critique.score:.2f}\n"
                f"Issues:\n" + "\n".join(f"- {i}" for i in critique.issues) + "\n\n"
                f"Suggestions:\n" + "\n".join(f"- {s}" for s in critique.suggestions) + "\n\n"
                "Please produce an improved draft that addresses all the issues above."
            )
            messages.append(HumanMessage(content=revision_prompt))

        response = model.invoke(messages)
        new_draft = response.content.strip() 

        print(f"\n── Generated New Draft ───────────────────────────")
        cprint(f"=> {new_draft}", C.YELLOW)
        print(f"\n──────────────────────────────────────────────────")


        state.draft = new_draft
        state.iteration += 1
        state.messages.append(response)
        return state
        

    return drafter_node

