
from pydantic import BaseModel
import uuid


class RAGIngestionPayloadState(BaseModel):
    data: str
    event_id: uuid.UUID
    session_id: uuid.UUID
    project_id: uuid.UUID