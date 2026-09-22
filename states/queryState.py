
from typing import Annotated, List
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agentic_ai_platform.states.evaluation_state import EvaluationState
from agentic_ai_platform.states.query_rewritting_state import QueryRewritting




class QueryState(BaseModel):
    query: Annotated[str, Field(default="", description="The original user query that needs to be rewritten.")]    

    documents: Annotated[List[Document]| None, Field(default_factory=list, description="The retrieved documents relevant to the query.")]

    evaluation: Annotated[EvaluationState | None, Field(default=None, description="The evaluation results for the rewritten query.")]

    generation: Annotated[str | None, Field(default="", description="The final generated answer based on the rewritten query and retrieved documents.")]

    rewriteQueryState : Annotated[QueryRewritting, Field(default=None, description="state once analasys the original query to determine if it needs to be rewritten")]
