import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agentic_ai_platform.prompt_storage.prompt_registry import PromptRegistry

prompt_hub = PromptRegistry()

#weaviate = WeaviateDB()