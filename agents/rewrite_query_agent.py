import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type

from agentic_ai_platform.states.queryState import QueryState
from agentic_ai_platform.llm.llm import LLM




def create_rewrite_agent(schema: Type[BaseModel],
                         llm: LLM,
                         system_prompt: str):


    def rewrite_query_agent(state:BaseModel):
        """
        Rewrite the input query to improve its clarity and specificity.

        Args:
            state (QueryState): The current query state.
        Returns:
            str: The rewritten query.
        """

        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Rewrite the following query to improve its clarity and specificity and returns the rewritten query:\n\n{original_query}")
        ]).format_messages(
            original_query=state.query_state.question
        )

        
        rewritten_query = llm.invoke(prompt)
        rewritten_query = re.search(r'"([^"]*)"',rewritten_query.content).group(1) # extract the rewritten query from the LLM response, assuming it's enclosed in quotes

        state.query_state.rewritten_question = rewritten_query
        state.plan.input = rewritten_query
        return state.model_copy(update={"query_state": state.query_state, 
                                        "plan": state.plan})
        

    return rewrite_query_agent
    



    