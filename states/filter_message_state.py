
from typing import List
from pydantic import Field, BaseModel


class FilterMessageItem(BaseModel):
    index: int = Field(..., description="the index of the message in the input list")
    isIssued : bool = Field(..., description="True if the messages seems to mention about issues otherwise False")
    reasoning: str = Field(..., description="the reasoning behind why the message is relevant to any issue they found")
    cleaned_message : str = Field(..., description="original message")
    #is_relevant : bool = Field(..., description="True if the message mentioned any issue they found otherwise False")


class FilterMessageBatchState(BaseModel):
    items: List[FilterMessageItem] = Field(default_factory=list, description="classification result for each input message, one item per message")
