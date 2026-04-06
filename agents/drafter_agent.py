from typing import Type
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.state_manager.tool_state import ToolState
from agentic_ai_platform.utils.color_print import C, cprint


def create_drafter_agent(schema: Type[BaseModel],
                         tool_llm=None,
                         graph_llm=None,
                         tools=None):
    """
    Factory that returns a drafter node bound to the given Pydantic schema.

    The returned node produces drafts for the given task, incorporating any critic feedback on subsequent iterations.

    Args:
        schema: The Pydantic model class representing the state. Must include `task`, `draft`, `iteration`, and `critique` fields.
                Default is DraftState, but you can use a custom model as long as it has those fields.
    """
    def drafter_node(state: DraftState) -> DraftState:
        

        messages = []

        system_text = state.system_prompt
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

        tool_llm_chain = tool_llm.bind_tools(tools)
        response = tool_llm_chain.invoke("I would like to get any information for Rivian")

        if hasattr(response, "tool_calls") and response.tool_calls:
            # If the LLM called any tools, we assume the final draft is in the last tool call's output
            for call in response.tool_calls:
                cprint(f"\n── Tool Called: {call['name']} ──────────────────────", C.MAGENTA)
                cprint(f"=> Input:\n{call['args']}", C.MAGENTA)                

                if call['name'] in [k['tool_name'] for k in state.tool_calls]:
                    state.tool_calls.pop( state.tool_calls.index(call['args']))

                tool_result = tools[0].invoke(call['args'])  # Call the tool with the provided arguments
                state.tool_calls.append(ToolState(query= call['args']['query'], 
                                                  tool_name=call['name'],
                                                  tool_args=call['args'], 
                                                  tool_result=tool_result))

                if tool_result is None or (isinstance(tool_result, str) and tool_result.strip() == ""):
                    cprint( f"No result returned from tool({call['name']}) => {call['args']}", C.MAGENTA)
                else:
                    cprint(f"=> Output:\n{tool_result}", C.MAGENTA)
                print(f"──────────────────────────────────────────────────")

            

        response = graph_llm.invoke(messages)
        new_draft = response.content.strip() 

        print(f"\n── Generated New Draft ───────────────────────────")
        cprint(f"=> {new_draft}", C.YELLOW)
        print(f"\n──────────────────────────────────────────────────")


        state.draft = new_draft
        state.iteration += 1
        state.messages.append(response)
        return state
        

    return drafter_node

