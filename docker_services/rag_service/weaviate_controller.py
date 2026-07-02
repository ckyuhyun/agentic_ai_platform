import os
import uuid
from typing import Optional, List, Dict, Union
from dotenv import load_dotenv
import weaviate
import weaviate.classes.config as wvc
from langchain_weaviate.vectorstores import WeaviateVectorStore


from weaviate.classes.config import Configure, VectorDistances 
import json
import logging
import sys

from agentic_ai_platform.data.weaviate_property_data import WeaviateProperty
from agentic_ai_platform.enum.weaviate_status_code import WeaviateStatusCode


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

class WeaviateController:
    def __init__(self,
                 collection_name: Optional[str] = None, 
                 embedded_model: Optional[str] = None):
        
        load_dotenv()


        self._collection_name_ = collection_name
        self._embedded_model = embedded_model
        self._properties_config = None

        self._api_endpoint_host_ = os.getenv("WEAVIATE_HOST")
        self._api_endpoint_port_ = os.getenv("WEAVIATE_PORT")

        #self.__collection_init__()

    @property
    def collection_name(self):
        return self._collection_name_

    @property
    def embedded_model(self):
        return self._embedded_model
    
    @property
    def properties_config(self):
        return self._properties_config
        

    @collection_name.setter
    def collection_name(self,
                       value:str):
        self._collection_name_ = value

    @embedded_model.setter
    def embedded_model(self,
                       value:str):
        self._embedded_model = value

    @properties_config.setter
    def properties_config(self, 
                   value : list):
        self._properties_config = value

    

    @property
    def db_connected(self) -> bool:
        """Check whether the configured Weaviate instance is reachable."""
        try:
            with weaviate.connect_to_local(host=self._api_endpoint_host_,
                                           port=self._api_endpoint_port_) as client:
                return client.is_ready()
        except Exception:
            return False

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

        if self._collection_name_ is None:
            raise Exception("collection name not addressed")
        
        if self._properties_config is None:
            raise Exception("properties not being set yet")
               
            

        with weaviate.connect_to_local(host=self._api_endpoint_host_, 
                                       port=self._api_endpoint_port_) as client: 
            
            try:
                if client.collections.exists(self._collection_name_):
                    client.collections.delete(self._collection_name_)
            except Exception as e:
                raise Exception(f"Error deleting existing collection: {str(e)}")

            if not client.collections.exists(self._collection_name_):
                wvc_property = []
                # for p in self._properties:
                #     wvc_property.append(
                #         wvc.Property(
                #             name=p.name,
                #             data_type=p.datatype,
                #             description=p.description
                #         )
                #     )

                try:
                    client.collections.create(
                        name=self._collection_name_,
                        description=f"Collection for {self._collection_name_}",
                        properties=self._properties_config,                        
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                            ef_construction=128, # Build-time accuracy (higher = better, slower)
                            max_connections=64, # Graph connectivity
                            ef=64 # Query-time accuracy
                        )

                    )
                except Exception as e:
                    raise Exception(f"Error creating collection: {str(e)}")
                
            collection_object =  client.collections.get(self._collection_name_)

            # weaviate_vector_logger = logging.getLogger("langchain_weaviate.vectorstores")

            # # set its level to DEBUG 
            # weaviate_vector_logger.setLevel(logging.DEBUG)

            # handler = logging.StreamHandler(sys.stdout)
            # handler.setLevel(logging.DEBUG)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # handler.setFormatter(formatter)

            # weaviate_vector_logger.addHandler(handler)
            # weaviate_vector_logger.propagate = False

            #db = self.__get_weaviate_db__(client)

            # try:
            #     print(f"was trying to add {len(text_documents)} documents.")
            #     result = db.add_documents(documents=text_documents,
            #                               ids=)
            #     print(f"Added {len(result)} documents.")

            # except Exception as e:
            #     raise Exception(f"Error adding documents to Weaviate: {str(e)}")
            try:
                if text_documents:
                    #create a new uuid
                    object_uuid = str(uuid.uuid4())

                    if batch_size:
                        with collection_object.batch.fixed_size(batch_size=batch_size) as batch:
                            if isinstance(text_documents, list):
                                for obj in text_documents:
                                    # ensure each object has a uuid
                                    if isinstance(obj, dict):
                                        obj.setdefault("uuid", object_uuid)

                                    batch.add_object(properties=obj,
                                                     uuid=object_uuid)
                            elif isinstance(text_documents, dict):
                                text_documents.setdefault("uuid", object_uuid)
                                batch.add_object(properties=text_documents,
                                                 uuid=object_uuid)
                    else:
                        with collection_object.batch.dynamic() as batch:    
                            if isinstance(text_documents, list):
                                for obj in text_documents:
                                    #if isinstance(obj.model_fields, dict):
                                    #    obj.model_fields.setdefault("uuid", object_uuid)

                                    p = obj.page_content if hasattr(obj, 'page_content') else obj
                                    

                                    batch.add_object(properties= dict(content=p),
                                                     uuid=object_uuid)
                                    
                            elif isinstance(text_documents, dict):
                                text_documents.setdefault("uuid", object_uuid)
                                batch.add_object(properties=text_documents,
                                                 uuid=object_uuid)
            except Exception as e:
                raise Exception(f"Error updating collection: {str(e)}")
            

    def read_data(self) -> None:
        if self._collection_name_ is None:
            return WeaviateStatusCode.NoCollection
        
        with weaviate.connect_to_local(host=self._api_endpoint_host_, 
                                       port=self._api_endpoint_port_) as client:
            if client.collections.exists(self._collection_name_):
                collection_object =  client.collections.use(self._collection_name_)
                for item in collection_object.iterator():
                    print("---")
                    print(f"item: {item}")

        
    def search_query(
            self,
            query: str,
            top_k: int = 5) -> Union[List[Dict], WeaviateStatusCode, None]:
        
        if self._collection_name_ is None:
            return WeaviateStatusCode.NoCollection
        
        with weaviate.connect_to_local(host=self._api_endpoint_host_,
                                       port=self._api_endpoint_port_) as client:
            
            if client.collections.exists(self._collection_name_):
                try:
                    db = self.__get_weaviate_db__(client)
                    return db.similarity_search(query, k=top_k)
                except Exception as e:
                    raise(e)
            return WeaviateStatusCode.NoExistCollection
    
    def delete_collection(self):
        if self._collection_name_ is None:
            return WeaviateStatusCode.NoCollection
        
        with weaviate.connect_to_local(host=self._api_endpoint_host_, 
                                       port=self._api_endpoint_port_) as client:
            client.collections.delete(self._collection_name_)

    # def __collection_init__(self):
    #         with weaviate.connect_to_local(host=self.api_endpoint_host, 
    #                                        port=self.api_endpoint_port) as client:
    #             try:
    #                 if not client.collections.exists(self._collection_name):                    
    #                     client.collections.create(
    #                         self._collection_name,
    #                         vector_config=Configure.Vectors.text2vec_ollama()
    #                     )            
    #             except Exception as e: 
    #                 raise Exception(f"Error initializing collection: {str(e)}")
                
    def __get_weaviate_db__(self,
                         client):
        return WeaviateVectorStore(client,
                                         index_name=self._collection_name_,
                                         text_key="text",
                                         embedding=self.embedded_model)      
        
