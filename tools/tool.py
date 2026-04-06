from typing import Union
from langchain_core.tools import tool
from pydantic import ValidationError
from agentic_ai_platform.rag.weaviate_controller import WeaviateController
from agentic_ai_platform.state_manager.draft_state import DraftState
from agentic_ai_platform.state_manager.tool_state import ToolState

class Tools:
    @staticmethod
    @tool
    def tool_a():
        """Example tool A"""
        print("Executing Tool A")
        return "Result from Tool A"

    @staticmethod
    @tool    
    def search_rag(query:str) -> Union[None, str, list]:
        """Search the RAG system with the given query and return results."""
        result = WeaviateController().search_query(query)

        return None if not result else result