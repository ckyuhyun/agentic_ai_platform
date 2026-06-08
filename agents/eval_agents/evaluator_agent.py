
from typing import Type

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate


def create_evaluator_agent(
    schema: Type[BaseModel],
    system_prompt: str = None,
    graph_llm  = None,
):
    """
    Factory that returns a evaluator node bound to the given Pydantic schema.

    schema must have: evaluation (str).
    """
    

    def evaluator_agent(state):
        prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")]).format_messages(
                    input=state.plan.input
                )
        
        structed_model = graph_llm.with_structured_output(schema)
        
        evaluation_result = structed_model.invoke(prompt)

        updated_query_state = state.query_state.model_copy(update={"evaluation": evaluation_result})

        return state.model_copy(update={"query_state": updated_query_state})

    return evaluator_agent