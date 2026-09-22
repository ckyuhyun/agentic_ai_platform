from pydantic import BaseModel, Field
from typing import Annotated

class QueryRewritting(BaseModel):
    is_ambiguous: Annotated[bool,  Field(
        description="""
        False only if the query is missing information needed to plan that can't be reasonably defaulted
        """
    )]
    is_relevant: Annotated[bool,  Field(
            description="""
            True only if the query sounds like work, team or project
            """
        )]
    reason : Annotated[str, Field(default="", 
                                  description="Explain why it is not relevant or ambiguous with mentioning what the reasoning is for")]

    rewritten_query: Annotated[str, Field(default="",
                                          description="The query rewritten to be unambiguous, filling in reasonable defaults when possible -- "
                                                      "leave blank if is_ambiguous and no reasonable default exists"
    )]
    clarifying_question: Annotated[str, Field(
        default="",
        description="If is_ambiguous and no reasonable default exists, the question to ask the human -- otherwise blank"
    )]
