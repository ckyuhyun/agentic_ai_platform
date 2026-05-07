from typing import Optional, List, Dict, Union
import weaviate

from weaviate.classes.config import Configure 
import json


# /c:/Users/c_kyu/Development/PersonalProject/rag_project/db/weaviate.py


# def get_weaviate_client(
#     url: str = "http://localhost:8080",
#     api_key: Optional[str] = None,
#     additional_headers: Optional[Dict[str, str]] = None,
# ) -> weaviate.Client:
#     """
#     Create and return a weaviate.Client.
#     - url: weaviate endpoint (e.g. "http://localhost:8080")
#     - api_key: optional API key (will be added as X-OpenAI-Api-Key by default)
#     - additional_headers: any other headers to include
#     """
#     headers = dict(additional_headers or {})
#     if api_key:
#         # common header key for OpenAI-backed modules; adjust if your setup needs a different header
#         headers.setdefault("X-OpenAI-Api-Key", api_key)
#     return weaviate.Client(url=url, additional_headers=headers)

class WeaviateDB:
    def __init__(self):
        self.api_endpoint = "http://ollama:11434"
        self.collection_name = "Ollama_Models"
        self.__collection_init__()

    
  
            
    
    def update_query(
        self,
        data_objects: Union[List[Dict], Dict] = None) -> None:
        with weaviate.connect_to_local() as client:
            collection_object =  client.collections.use(self.collection_name)

            if data_objects:
                with collection_object.batch.fixed_size(batch_size=200) as batch:    
                    if isinstance(data_objects, list):
                        for obj in data_objects:
                            batch.add_object(properties=obj)
                    elif isinstance(data_objects, dict):
                        batch.add_object(properties=data_objects)

    def read_data(self) -> None:
        with weaviate.connect_to_local() as client:
            if client.collections.exists(self.collection_name):
                collection_object =  client.collections.use(self.collection_name)
                for item in collection_object.iterator():
                    print("---")
                    print(f"item: {item}")

                    
    def search_query(
            self,
            query_text: str,
            limit: int = 5) -> Union[List[Dict], Dict, None]:
        
        with weaviate.connect_to_local() as client:
            search_results = []
            if client.collections.exists(self.collection_name):
                collection_object =  client.collections.use(self.collection_name)
                # result = collection_object.query.near_text(
                #     query=query_text,
                #     limit=limit
                # )
                result = collection_object.query.fetch_objects()
                for obj in result.objects:
                    search_results.append(obj.properties)
                    print(json.dumps(obj.properties, indent=2))  # Inspect the results
                
                return search_results
            return None
    
    def delete_collection(self):
        with weaviate.connect_to_local() as client:
            client.collections.delete(self.collection_name)

    def __collection_init__(self):
            with weaviate.connect_to_local() as client:
                if not client.collections.exists(self.collection_name):                    
                    client.collections.create(
                        self.collection_name,
                        #vector_config=Configure.Vectors.text2vec_ollama()
                    )            
        
