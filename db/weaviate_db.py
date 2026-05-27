import os
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
import weaviate
import weaviate.classes.config as wvc
from langchain_weaviate.vectorstores import WeaviateVectorStore


from weaviate.classes.config import Configure 
import json
import logging
import sys


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
    def __init__(self,
                 collection_name: str,
                 embedded_model):
        
        load_dotenv()


        self.collection_name = collection_name
        self.api_endpoint_host = os.getenv("WEAVIATE_HOST")
        self.api_endpoint_port = os.getenv("WEAVIATE_PORT")
        self.embedded_model = embedded_model
        self.__collection_init__()
            
    
    def update_query(
        self,
        text_documents : Union[List[Dict], Dict],
        batch_size : Union[int, None] = None,
        
        #emb_objects: Union[List[Dict], Dict] = None) -> None:
         ) -> None:
        """
        Updates the collection with new data objects. Can be done in batches or all at once.
        - batch_size: if provided, data will be added in batches of this size; otherwise, all data will be batched dynamically.
        - emb_objects: a single dict or a list of dicts representing the objects to add to the collection. 
                        Each dict should contain the properties of the object to be added.  
        """        

      
        with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                       port=self.api_endpoint_port) as client:              
            try:
                if client.collections.exists(self.collection_name):
                    client.collections.delete(self.collection_name)
            except Exception as e:
                raise Exception(f"Error deleting existing collection: {str(e)}")

        with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                       port=self.api_endpoint_port) as client: 

            if not client.collections.exists(self.collection_name):
                try:
                    client.collections.create(
                        name=self.collection_name,
                        description=f"Collection for {self.collection_name}",
                        
                        properties=[
                            wvc.Property(
                                name="page_content",
                                data_type=wvc.DataType.TEXT,
                                description="The text content of the document"
                            ),
                            # wvc.Property(
                            #     name="coordinates",
                            #     data_type=wvc.DataType.TEXT, 
                            #     index_filterable=False,
                            #     index_searchable=False,
                            #     description="Raw layout metadata ignored by vector search indices"
                            # ),
                #             wvc.Property(
                #                 name="points",
                #                 data_type=wvc.DataType.TEXT, 
                # #                index_filterable=False,
                #  #               index_searchable=False,
                #                 description="Raw layout metadata ignored by vector search indices"
                #             ),
                            # wvc.Property(
                            #     name="metadata",
                            #     data_type=wvc.DataType.TEXT,
                            #     description="Metadata associated with the document"
                            # )

                        ],
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=wvc.Configure.Vectors.Distance.COSINE,
                            ef_construction=128, # Build-time accuracy (higher = better, slower)
                            max_connections=64, # Graph connectivity
                            ef=64 # Query-time accuracy
                        )

                    )
                except Exception as e:
                    raise Exception(f"Error creating collection: {str(e)}")
                
            collection_object =  client.collections.get(self.collection_name)

            # weaviate_vector_logger = logging.getLogger("langchain_weaviate.vectorstores")

            # # set its level to DEBUG 
            # weaviate_vector_logger.setLevel(logging.DEBUG)

            # handler = logging.StreamHandler(sys.stdout)
            # handler.setLevel(logging.DEBUG)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # handler.setFormatter(formatter)

            # weaviate_vector_logger.addHandler(handler)
            # weaviate_vector_logger.propagate = False

            db = WeaviateVectorStore(client, 
                                     index_name=self.collection_name,
                                     text_key="text",
                                     embedding=self.embedded_model)
            try:
                result = db.add_documents(documents=text_documents)
            except Exception as e:
                raise Exception(f"Error adding documents to Weaviate: {str(e)}")
            # try:
            #     if text_documents:
            #         #create a new uuid
            #         object_uuid = str(uuid.uuid4())

            #         if batch_size:
            #             with collection_object.batch.fixed_size(batch_size=batch_size) as batch:                                
            #                 if isinstance(text_documents, list):
            #                     for obj in text_documents:
            #                         # ensure each object has a uuid
            #                         if isinstance(obj, dict):
            #                             obj.setdefault("uuid", object_uuid)

            #                         batch.add_object(properties=obj,
            #                                          uuid=object_uuid)
            #                 elif isinstance(text_documents, dict):
            #                     text_documents.setdefault("uuid", object_uuid)
            #                     batch.add_object(properties=text_documents,
            #                                      uuid=object_uuid)
            #         else:
            #             with collection_object.batch.dynamic() as batch:    
            #                 if isinstance(text_documents, list):
            #                     for obj in text_documents:
            #                         if isinstance(obj, dict):
            #                             obj.setdefault("uuid", object_uuid)
            #                         batch.add_object(properties=obj,
            #                                          uuid=object_uuid)
                                    
            #                 elif isinstance(text_documents, dict):
            #                     text_documents.setdefault("uuid", object_uuid)
            #                     batch.add_object(properties=text_documents,
            #                                      uuid=object_uuid)
            except Exception as e:
                raise Exception(f"Error updating collection: {str(e)}")
            

    def read_data(self) -> None:
        with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                       port=self.api_endpoint_port) as client:
            if client.collections.exists(self.collection_name):
                collection_object =  client.collections.use(self.collection_name)
                for item in collection_object.iterator():
                    print("---")
                    print(f"item: {item}")

                    
    def search_query(
            self,
            query_text: str,
            limit: int = 5) -> Union[List[Dict], Dict, None]:
        
        with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                       port=self.api_endpoint_port) as client:
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
        with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                       port=self.api_endpoint_port) as client:
            client.collections.delete(self.collection_name)

    def __collection_init__(self):
            with weaviate.connect_to_local(host=self.api_endpoint_host, 
                                           port=self.api_endpoint_port) as client:
                try:
                    if not client.collections.exists(self.collection_name):                    
                        client.collections.create(
                            self.collection_name,
                            vector_config=Configure.Vectors.text2vec_ollama()
                        )            
                except Exception as e: 
                    raise Exception(f"Error initializing collection: {str(e)}")
        
