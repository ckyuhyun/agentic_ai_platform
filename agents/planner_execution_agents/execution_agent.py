
from typing import List, Type
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agentic_ai_platform.state_manager.supervise_state import NodeTrace
from agentic_ai_platform.state_manager.tool_state import ToolState


def create_execution_agent(
    system_prompt: List,
    graph_llm=None,
    tool_llm=None,
    tools=None,
):
    """
    Factory that returns an execution node that iterates over the plan steps
    and invokes tools for each step if applicable.
    """

    def execution_agent(state):
        trace = NodeTrace.start(node="executor", iteration=state.iteration, model="llama3.1")

        tools_invoked: List[str] = []
        past_steps: List[str] = list(state.plan.past_steps)
        plan_steps = state.plan.plan

        # check if the plan_steps has correct dictionary key.
        if plan_steps is not None  and \
            not isinstance(plan_steps[0], dict)  and \
            not all([lambda k: "description" in k or "tool" in k or  "parameters" in k for k in plan_steps[0].keys()]):
            raise ValueError("Each step in the plan must be a dictionary with at least a 'description' key.")

        # Resume from where we left off
        for step in plan_steps[len(past_steps):]:
            step_description = step.get("description", str(step))
            tool_hint = step.get("tool")
            tool_parameters = step.get("parameters", {})

            # Build context messages for this step
            messages = []
            if system_prompt:
                messages.extend(ChatPromptTemplate.from_messages(system_prompt).format_messages())

            # context = f"Original task: {state.plan.input}\n\nCompleted steps so far:\n"
            # for ps in past_steps:
            #     context += f"- {ps}\n"
            # messages.append(HumanMessage(
            #     content=f"{context}\nNow execute this step: {step_description}"
            # ))

            result_text = None

            #1. If the plan step already names a tool + args, call it directly
            if tool_hint and tools and tool_parameters:
                matching_tool = next((t for t in tools if t.name == tool_hint), None)
                if matching_tool:
                    tool_result = matching_tool.invoke(tool_parameters)
                    tools_invoked.append(tool_hint)
                    state.tool_calls.append(ToolState(
                        query=tool_parameters.get("query", step_description),
                        tool_name=tool_hint,
                        tool_args=tool_parameters,
                        tool_result=tool_result,
                    ))
                    result_text = str(tool_result)

            # 2. Let the LLM decide which tool to call for this step
            if result_text is None and tool_llm and tools:
                tool_bound_llm = tool_llm.bind_tools(tools)
                final_query = state.query_state.rewritten_question
                response = tool_bound_llm.invoke(final_query)

                if hasattr(response, "tool_calls") and response.tool_calls:
                    for call in response.tool_calls:
                        matching_tool = next((t for t in tools if t.name == call["name"]), None)
                        if matching_tool:
                            tool_result = matching_tool.invoke(call["args"])
                            tools_invoked.append(call["name"])
                            state.tool_calls.append(ToolState(
                                query=call["args"].get("query", step_description),
                                tool_name=call["name"],
                                tool_args=call["args"],
                                tool_result=tool_result,
                            ))
                            result_text = (result_text or "") + str(tool_result) + "\n"

            # 3. Fallback: pure LLM reasoning for steps that need no tool
            if result_text is None:
                final_query =  state.query_state.rewritten_question
                response = graph_llm.invoke(final_query)
                result_text = response.content.strip()

            past_steps.append(f"Step: {step_description}\nResult: {result_text.strip()}")

        state.plan.past_steps = past_steps
        state.iteration += 1
        state.node_traces.append(trace.finish(tool_calls_made=tools_invoked))

        return state

    return execution_agent
