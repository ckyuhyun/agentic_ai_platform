import logging

from agentic_ai_platform.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from agentic_ai_platform.prompt_storage.prompt_registry import PromptRegistry

prompt_hub = PromptRegistry()

#weaviate = WeaviateDB()