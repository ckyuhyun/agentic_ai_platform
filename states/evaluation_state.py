from pydantic import BaseModel, Field


class EvaluationState(BaseModel):
    accuracy_score: int = Field(
        ...,
        description="Score from 1-5 on factual accuracy.")
    
    completeness_score: int = Field(
        ...,
        description="Score from 1-5 on addressing all parts of the prompt.")
    
    tone_score: int = Field(
        ...,
        description="Score from 1-5 on appropriate formatting and tone.")
    
    reasoning: str = Field(
        ...,
        description="Detailed explanation for the given scores.")
    
    passed: bool = Field(
        ...,
        description="True if ALL aspects meet the minimum threshold (e.g., score >= 4).")

    intent_alignment_score: float = Field(
        ..., 
        description="Score 0.0 to 1.0. " \
                         "Does the rewrite maintain original intent without adding hallucinations? " \
                         "score from 0 (hallucinated) to 1 (not hallucinated)"
    )
    disambiguation_score: float = Field(
        ..., 
        description="Score 0.0 to 1.0. " \
                         "Did it resolve pronouns or inject necessary conversational context? " \
                         "score from 0 (ambiguous) to 1 (clear)"
    )
    retrieval_suitability_score: float = Field(
        ..., 
        description="Score 0.0 to 1.0. " \
                         "Is it stripped of conversational filler and optimized for keyword/vector search? " \
                         "score from 0 (not suitable) to 1 (suitable)"
    )
    suggested_rewrites: str = Field(
        description="Specific instructions on how to rewrite the prompt or tools to fix the issue." )
    
    user_suggestions: list[str] = Field(
        description="""list of questions to ask the user to clarify their intent or provide missing information in detail.
                       the goal of this suggestions is to get better plans and not to generate more user_suggestions for next time.
                       these should be written very clearly and user-friendly.
                       """
    )