
from typing import List
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agentic_ai_platform.states.evaluation_state import EvaluationState




class QueryState(BaseModel):
    question: str = Field(default="", description="The original user query that needs to be rewritten.")
    rewritten_question: str = Field(default="", description="The rewritten query after processing by the rewrite agent.")
    documents: List[Document] = Field(default_factory=list, description="The retrieved documents relevant to the query.")
    evaluation: EvaluationState = Field(default=None, description="The evaluation results for the rewritten query.")
    generation: str = Field(default="", description="The final generated answer based on the rewritten query and retrieved documents.")
    