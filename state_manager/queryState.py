
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

class QueryRewriterEvaluatorResult(BaseModel):
    intent_alignment_score: float = Field(
        ..., description="Score 0.0 to 1.0. " \
                         "Does the rewrite maintain original intent without adding hallucinations? " \
                         "score from 0 (hallucinated) to 1 (not hallucinated)"
    )
    disambiguation_score: float = Field(
        ..., description="Score 0.0 to 1.0. " \
                         "Did it resolve pronouns or inject necessary conversational context? " \
                         "score from 0 (ambiguous) to 1 (clear)"
    )
    retrieval_suitability_score: float = Field(
        ..., description="Score 0.0 to 1.0. " \
                         "Is it stripped of conversational filler and optimized for keyword/vector search? " \
                         "score from 0 (not suitable) to 1 (suitable)"
    )
    reasoning: str = Field(..., description="Concise justification explaining the assigned scores.")


class QueryState(BaseModel):
    question: str = Field(default="", description="The original user query that needs to be rewritten.")
    rewritten_question: str = "" # Field(default_value="", description="The rewritten query after processing by the rewrite agent.")
    documents: List[Document] = None #Field(default_factory=list, description="The retrieved documents relevant to the query.")
    evaluation: QueryRewriterEvaluatorResult = None #Field(default=None, description="The evaluation results for the rewritten query.")
    generation: str = "" #Field(default_value="", description="The final generated answer based on the rewritten query and retrieved documents.")
