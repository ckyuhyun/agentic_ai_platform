
from pydantic import BaseModel
import uuid


class RAGIngestionPayloadState(BaseModel):
    data: str
    event_id: uuid.UUID
    thread_id: uuid.UUID
    