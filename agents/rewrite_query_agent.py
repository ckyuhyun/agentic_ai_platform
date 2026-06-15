import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Type

from agentic_ai_platform.states.queryState import QueryState
from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.states.supervise_state import SuperviseState
from agentic_ai_platform.utils.message_utils import extract_new_messages




def create_rewrite_agent(schema: Type[BaseModel],
                         llm: LLM,
                         system_prompt: str):


    def rewrite_query_agent(state:SuperviseState):
        """
        Rewrite the input query to improve its clarity and specificity.

        Args:
            state (QueryState): The current query state.
        Returns:
            str: The rewritten query.
        """

        new_messages, new_message_index = extract_new_messages(state)
        human_feedback = "\n".join(
            msg.content for msg in new_messages if isinstance(msg, HumanMessage)
        )

        if human_feedback:
            user_prompt = (
                "Rewrite the following query to improve its clarity and specificity, "
                "taking the human clarifications below into account, and return the rewritten query:\n\n"
                "Original query:\n{original_query}\n\n"
                "Human clarifications:\n{human_feedback}"
            )
        else:
            user_prompt = "Rewrite the following query to improve its clarity and specificity and returns the rewritten query:\n\n{original_query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ]).format_messages(
            original_query=state.plan.input,
            human_feedback=human_feedback
        )


        rewritten_query = llm.invoke(prompt)
        #rewritten_query = re.search(r'"([^"]*)"',rewritten_query.content).group(1) # extract the rewritten query from the LLM response, assuming it's enclosed in quotes

        state.query_state.rewritten_question = rewritten_query.content
        #state.plan.input = rewritten_query
        return state.model_copy(update={"query_state": state.query_state,
         #                               "plan": state.plan,
                                        "last_reviewed_message_index": new_message_index})
        

    return rewrite_query_agent
    



     