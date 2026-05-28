from agentic_ai_platform.state_manager.queryState import QueryState
from llm.llm import LLM

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type


def create_rewrite_agent(schema: Type[BaseModel],
                         llm: LLM,
                         ):


    def rewrite_query_agent(state : QueryState):
        """
        Rewrite the input query to improve its clarity and specificity.

        Args:
            state (QueryState): The current query state.
        Returns:
            str: The rewritten query.
        """

        original_query = state.query
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that helps rewrite user queries to be more clear and specific."),
            ("user", f"Rewrite the following query to improve its clarity and specificity:\n\n{original_query}")
        ])

        
        rewritten_query = llm.invoke(prompt.format_messages())

        state.rewritten_query = rewritten_query
        return state
        

    return rewrite_query_agent
    



    