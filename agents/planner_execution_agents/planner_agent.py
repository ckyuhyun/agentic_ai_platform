from typing import List, Type, Union
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig


from agentic_ai_platform.states.supervise_state import NodeTrace
from agentic_ai_platform.states.plan_state import PlanState



def create_planner_agent(
    schema: Type[BaseModel],
    system_prompt: str = None,
    graph_llm  = None,
):
    """
    Factory that returns a planner node bound to the given Pydantic schema.

    schema must have: plan (str), reasoning (str), next_steps (list[str]).
    """
    

    def planner_agent(state):
        
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")]).format_messages(
                    input=state.query_state.rewritten_question
                )
        
        trace = NodeTrace.start(node="planner", 
                                iteration=state.iteration, 
                                model="llama3.1")

        structed_model = graph_llm.with_structured_output(schema)\
                                  .with_config(RunnableConfig(
                                            metadata={
                                                "node": "planner", 
                                                "iteration": state.iteration, 
                                                "model": trace.model},
                                            tags=["planner"]
                                        ))
        
        planstate: PlanState = structed_model.invoke(prompt)        
        
        return state.model_copy(update={"plan": planstate})

    return planner_agent
