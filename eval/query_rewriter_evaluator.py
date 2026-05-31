from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agentic_ai_platform.state_manager.queryState import QueryRewriterEvaluatorResult


class QueryRewriterEvaluator:
    def __init__(self,
                 system_eval_prompt:str):        
        self.system_eval_prompt = system_eval_prompt

    def evaluate(self,
                 query: str,
                 LLM):
        """
        Evaluates the performance of the query rewriter by comparing its output to the expected rewrite.
        """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_eval_prompt),
            ("user", f"Original query:\n{query}\n\n"
                     "Evaluate the quality of the rewritten query.")
        ]).invoke({"query": query})

        structured_llm = LLM.with_structured_output(QueryRewriterEvaluatorResult)
        output =  structured_llm.invoke(eval_prompt)

        return output

         


        