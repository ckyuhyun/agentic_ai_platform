import re

from agentic_ai_platform.eval.query_rewriter_evaluator import QueryRewriterEvaluator
from agentic_ai_platform.state_manager.queryState import QueryState
from agentic_ai_platform.llm.llm import LLM

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type


def create_rewrite_agent(schema: Type[BaseModel],
                         llm: LLM,
                         system_prompt: str,
                         system_eval_prompt: str):


    def rewrite_query_agent(state):
        """
        Rewrite the input query to improve its clarity and specificity.

        Args:
            state (QueryState): The current query state.
        Returns:
            str: The rewritten query.
        """

        original_query = state.query_state.question
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"Rewrite the following query to improve its clarity and specificity and returns the rewritten query:\n\n{original_query}")
        ]).format_messages()

        
        rewritten_query = llm.invoke(prompt)
        rewritten_query = re.search(r'"([^"]*)"',rewritten_query.content).group(1) # extract the rewritten query from the LLM response, assuming it's enclosed in quotes

        rewrite_evalulator = QueryRewriterEvaluator(system_eval_prompt=system_eval_prompt)
        
        evaulated_result = rewrite_evalulator.evaluate(
            query=rewritten_query,
            LLM=llm
        )

        state.query_state.rewritten_question = rewritten_query
        state.query_state.evaluation = evaulated_result
        return state
        

    return rewrite_query_agent
    



    