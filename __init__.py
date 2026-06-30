from agentic_ai_platform.prompt_storage.prompt_registry import PromptRegistry
from agentic_ai_platform.rag_service.weaviate_db import WeaviateDB

prompt_hub = PromptRegistry()
weaviate = WeaviateDB()