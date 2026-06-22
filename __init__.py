from agentic_ai_platform.prompt_storage.prompt_registry import PromptRegistry
from agentic_ai_platform.db.weaviate_db import WeaviateDB

prompt_hub = PromptRegistry()
weaviate = WeaviateDB()