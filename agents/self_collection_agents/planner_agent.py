from typing import List, Type, Union
from langchain.messages import HumanMessage
from pydantic import BaseModel

from agentic_ai_platform.state_manager.draft_state import NodeTrace
from agentic_ai_platform.state_manager.plan_state import PlanState



def create_planner_agent(
    schema: Type[BaseModel],
    system_prompt: List,
    graph_llm  = None,
):
    """
    Factory that returns a planner node bound to the given Pydantic schema.
 
    schema must have: plan (str), reasoning (str), next_steps (list[str]).
    """

    def planner_agent(state):
        
        """
        This agent is responsible for creating a plan with a complex query requiring various actions for the next steps to take, given the current state of the draft. 
        It should return a structured output that includes the plan, reasoning, and next steps. 
        The queries being broken down would send to Executor agent to execute the plan and return the results, which would then be fed back to the planner for the next iteration.

        """

        # start updating trace
        trace = NodeTrace.start(node="planner", iteration=state.iteration, model="llama3.1")

        state.plan.input = state.task
        final_prompt = system_prompt + [HumanMessage(content=state.plan.input)]

        structed_model = graph_llm.with_structured_output(schema)
        output : PlanState = structed_model.invoke(final_prompt)

        return {'messages' : [output]}
        


        # structured_output = structured_model.invoke(prompt)

        # update trace with tool calls and LLM response
        # trace.add_tool_calls(tool_reports)
        # trace.set_response(structured_output)

        # return structured_output


        
    return planner_agent