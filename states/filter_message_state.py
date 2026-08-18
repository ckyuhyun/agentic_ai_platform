
from typing import Annotated, List
from pydantic import Field, BaseModel


class FilterMessageItemLLM(BaseModel):
    """
    Shape the model is asked to produce. Excludes cleaned_message so the model
    doesn't have to re-emit the full original text per item, which was blowing
    past small local models' completion-token budget on batched requests.
    """
    index: Annotated[int , Field(..., description="the index of the message in the input list")]
    scoring : Annotated[float, Field(...,
                                     ge=0.0,
                                     le=1.0,
                                     description="a score indicating the relevance of the message to any issues found. 0 means not relevant, 1 means highly relevant")]
    reasoning: Annotated[str, Field(..., description="the reasoning behind why the message is relevant to any issue they found")]


class FilterMessageBatchStateLLM(BaseModel):
    items: Annotated[List[FilterMessageItemLLM], Field(default_factory=list, description="classification result for each input message, one item per message")]


class FilterMessageItem(BaseModel):
    index: Annotated[int , Field(..., description="the index of the message in the input list")]
    scoring : Annotated[float, Field(...,
                                     ge=0.0,
                                     le=1.0,
                                     description="a score indicating the relevance of the message to any issues found. 0 means not relevant, 1 means highly relevant")]
    reasoning: Annotated[str, Field(..., description="the reasoning behind why the message is relevant to any issue they found")]
    cleaned_message : Annotated[str, Field(..., description="original message")]
    #is_relevant : bool = Field(..., description="True if the message mentioned any issue they found otherwise False")

