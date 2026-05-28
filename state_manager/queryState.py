
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class QueryState(BaseModel):
    question: str
    rewritten_question: str
    documents: List[Document]
    generation: str
