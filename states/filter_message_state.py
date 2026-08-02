
from typing import Annotated, List
from pydantic import Field, BaseModel


class FilterMessageItem(BaseModel):
    index: Annotated[int , Field(..., description="the index of the message in the input list")]
    scoring : Annotated[float, Field(..., 
                                     ge=0.0, 
                                     le=1.0, 
                                     description="a score indicating the relevance of the message to any issues found. 0 means not relevant, 1 means highly relevant")]
    reasoning: Annotated[str, Field(..., description="the reasoning behind why the message is relevant to any issue they found")]   
    cleaned_message : Annotated[str, Field(..., description="original message")]
    #is_relevant : bool = Field(..., description="True if the message mentioned any issue they found otherwise False")


class FilterMessageBatchState(BaseModel):
    items: Annotated[List[FilterMessageItem], Field(default_factory=list, description="classification result for each input message, one item per message")]
