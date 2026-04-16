from typing import Union, List
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import json

from agentic_ai_platform.rag.weaviate_controller import WeaviateController


_tavily = TavilySearch(max_results=5, topic="news", include_raw_content=False)


class Tools:
    @staticmethod
    @tool
    def tool_a():
        """Example tool A"""
        print("Executing Tool A")
        return "Result from Tool A"

    @staticmethod
    @tool
    def search_rag(query: str) -> Union[None, str, list]:
        """Search the RAG system with the given query and return results."""
        result = WeaviateController().search_query(query)
        return None if not result else result

    @staticmethod
    @tool
    def search_web(query: str) -> str:
        """Search the web for recent news and information about a given query.

        Args:
            query: The search query string 

        Returns:
            List of result dicts with keys: title, url, content, score.
        """
        results = _tavily.invoke({"query": query})
        filtered_list =  [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
                "score":   r.get("score", 0.0),
            }
            for r in (results if isinstance(results, list) else results.get("results", []))
        ]

        return json.dumps(filtered_list, ensure_ascii=False)