import asyncio
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings
from agentic_ai_platform.embedding.embedding_service import EmbeddingService


class LLamaIndexEmbeddings(BaseEmbedding):
    _emb_instance: Any = PrivateAttr()

    def __init__(self,
                 name: str,
                 bases: Any):
        super().__init__(model_name=name)
        self._emb_instance = bases()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._emb_instance.embed_text(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._emb_instance.embed_text(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # EmbeddingService is sync, so run the request off the event loop.
        return await asyncio.to_thread(self._emb_instance.embed_text, query)
