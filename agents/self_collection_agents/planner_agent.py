from typing import List, Type, Union
from langchain.messages import HumanMessage
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig


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

        trace = NodeTrace.start(node="planner", iteration=state.iteration, model="llama3.1")

        

        state.plan.input = state.task
        final_prompt = system_prompt + [HumanMessage(content=state.plan.input)]

        structed_model = graph_llm.with_structured_output(schema)\
                                  .with_config(RunnableConfig(
                                            metadata={
                                                "node": "planner", 
                                                "iteration": state.iteration, 
                                                "model": trace.model},
                                            tags=["planner"]
                                        ))
        
        output: PlanState = structed_model.invoke(
            final_prompt,
        )

        return {
            'messages': [output],
            'node_traces': state.node_traces + [trace],
        }


    return planner_agent
