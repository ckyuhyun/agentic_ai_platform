from typing import Type
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.state_manager.draft_state import DraftState, NodeTrace
from agentic_ai_platform.state_manager.tool_state import ToolState
from agentic_ai_platform.tools.tool import Tools
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
        trace = NodeTrace.start(node="drafter", iteration=state.iteration, model="llama3.1")

        messages = []
        messages.append(SystemMessage(content=state.system_prompt or ""))

        if state.iteration == 0 or state.critique is None:
            messages.append(HumanMessage(content=f"Task:\n{state.task}"))
        else:
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
        response = tool_llm_chain.invoke(messages)

        tools_invoked = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                tool_result = next(x for x in tools if x.name == call["name"]).invoke(call["args"])
                tools_invoked.append(call["name"])
                state.tool_calls.append(ToolState(
                    query=call["args"].get("query", ""),
                    tool_name=call["name"],
                    tool_args=call["args"],
                    tool_result=tool_result,
                ))

        response = graph_llm.invoke(messages)
        new_draft = response.content.strip()

        state.draft = new_draft
        state.iteration += 1
        state.messages.append(response)
        state.node_traces.append(
            trace.finish(tool_calls_made=tools_invoked, draft_len=len(new_draft))
        )
        return state
        

    return drafter_node

