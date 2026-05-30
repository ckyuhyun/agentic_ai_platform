
from typing import List,Type, Union
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

def create_execution_agent(
    schema: Type[BaseModel],
    system_prompt: List,
    graph_llm  = None,
):
    """
    Factory that returns an execution node bound to the given Pydantic schema.
    """

    def execution_agent(state):

        # final_prompt = system_prompt + [("human", "{input}")]
        # final_prompt = ChatPromptTemplate.from_messages(final_prompt).invoke({"input": state.plan.output})

        # structed_model = graph_llm.with_structured_output(schema)
        
        # output = structed_model.invoke(
        #     final_prompt,
        # )

        #for plan in state.plan:
        return state
            
        
    
    return execution_agent
    