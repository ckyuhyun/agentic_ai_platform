
from pydantic import BaseModel


class RAGIngestionPayloadState(BaseModel):
    content: str
    session_id: str
    project_id: str