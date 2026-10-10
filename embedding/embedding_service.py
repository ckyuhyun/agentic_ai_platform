import os
import requests
from typing import List, Union
import httpx


class EmbeddingService:
    """
    Generates embeddings via the TEI container's OpenAI-compatible API.

    Must stay on tei_embedding_model (whatever TEI is actually serving) since
    that's the model pgai's vectorizer-worker used to populate
    document_chunks_embedding. A different model/dimension here would make
    query vectors incomparable to the stored document vectors.
    """

    def __init__(self,
                 timeout: float = 10.0):
        """
        Initializes the EmbeddingService with the base URL of the embedding service and a timeout for requests.
        """

        self.base_url = os.getenv("TEI_EMBEDDING_URL").rstrip("/")
        self.embed_model = os.getenv('TEI_EMBEDDING_MODEL')
        self.timeout = timeout

    async def embed_text(self, 
                   text: str) -> Union[List[float],None]:
        embedded_texts = await self.embed_texts([text])
        
        return embedded_texts if embedded_texts else None

    async def embed_texts(self, 
                    texts: List[str]) -> List[float]:

        try:    
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=f"{self.base_url}/v1/embeddings",
                    json={"input": texts, "model": self.embed_model},
                    timeout=self.timeout,
                )
            response.raise_for_status()

            data = response.json()["data"]
            return [item["embedding"] for item in data][0]
            
        except Exception as e:
            print(f"Error: {e}")
            raise
            

        
