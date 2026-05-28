import os
from typing import List, Literal, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    UnstructuredFileLoader,
    DirectoryLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from agentic_ai_platform.db.weaviate_db import WeaviateDB
from agentic_ai_platform.graph.embedded_model_decision import EmbeddedModelDecision
from agentic_ai_platform.RAG.embedding import Embeddings
from agentic_ai_platform.RAG.embedded_model_list import EmbeddingModel


class Ingest:

    def __init__(self,
                 vector_db_type : Literal["weaviate"],
                 vector_db_collection_name: str,
                 chunk_size : int = 800,
                 chunk_overlap : int = 100):
        self.vector_db_type = vector_db_type
        self.vector_db_collection_name = vector_db_collection_name
        self.document_loader_split_enable : bool = False
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap,
                                                            length_function=len)



    def load_vector_store_from_text(self,
                                    text: Union[str, List[str]]) -> tuple[bool, str]:
        """
        Loads text data, chunks it, and builds a vector store.
        """

        try:
            documents = []
            if isinstance(text, str):
                documents.append([Document(page_content=text, metadata={})])
            elif isinstance(text, list):   
                documents.append([Document(page_content=t, metadata={}) for t in text])

            chunked_docs = self._get_chunk_documents_(documents)
            self._build_vector_store_(chunked_docs, 
                                    embedding_model=None)
            return True, "Vector store loaded successfully from text."
        except Exception as e:
            return False, f"Error loading vector store from text: {str(e)}"



        

    def load_vector_store_from_directory(self, 
                          directory_path:str, 
                          file_type: Literal["pdf", "txt", "docx"]) -> tuple[bool, str]:
        """
        Loads documents from a specified directory, chunks them, and builds a vector store.
        """
        try:
            documents = self._load_documents_from_directory_(directory_path, file_type)
            chunked_docs = self._get_chunk_documents_(documents)
            self._build_vector_store_(chunked_docs, 
                                    embedding_model=None)
            return True, "Vector store loaded successfully from directory."
        except Exception as e:
            return False, f"Error loading vector store from text: {str(e)}"

    def querying(self,  
                 query: str, 
                 top_k: int = 5) -> List[str]:
        """
        Queries the vector store with a given query and returns the results.
        """
        try:
            if self.vector_db_type == "weaviate":
                embed = Embeddings(internal_embedding_model=True,
                    embedding_model= None)
                        
                weaviate_db = WeaviateDB(collection_name=self.vector_db_collection_name, 
                                         embedded_model=embed.get_auto_decided_embedding_model())
                
                results = weaviate_db.search_query(query=query, 
                                                   top_k=top_k)
                return results
            else:
                raise ValueError(f"Unsupported vector_db_type: {self.vector_db_type}")
        except Exception as e:
            return False, f"Error querying vector store: {str(e)}"


    def _load_documents_from_directory_(self,
                                        directory_path:str, 
                                        file_type: Literal["pdf", "txt", "docx"]) -> List[Document]:
        """
        Loads documents from a specified directory based on the file type.
        """
        loaders = DirectoryLoader(path=directory_path, 
                                glob=f"**/*.{file_type}", 
                                loader_cls=UnstructuredFileLoader,
                                loader_kwargs= {"mode": "elements", 
                                                "include_metadata": True,
                                                "coordinates": False,
                                                "unstructured_kwargs": {"strategy": "fast"}} if self.document_loader_split_enable else None
                                )
        
        

        load_doc =  loaders.load()

        for doc in load_doc : doc.metadata.pop("coordinates", None) # remove coordinates from metadata if exists since we are not using it for vector search
        return load_doc


    def _get_chunk_documents_(self,
                              documents :List[Document]) -> List[Document]:
        """
        Chunks the documents into smaller pieces using RecursiveCharacterTextSplitter.
        """    

        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
        #                                             chunk_overlap=chunk_overlap,
        #                                             length_function=len)

        # if the document loader already splits the document, we can skip the splitting process
        if self.document_loader_split_enable:
            return documents
        
        split_docs = None
        try:
            split_docs = self._get_splitted_document_(documents)
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")

        
        return split_docs

    def _get_splitted_document_(self, 
                              document :Document) -> List[Document]:
        
        try:
            #doc = document[0] if isinstance(document, tuple) else document
            return self.text_splitter.split_documents(document) if isinstance(document, Document) else self.text_splitter.split_documents(document[0])
        except Exception as e:
            raise Exception("Error splitting document: " + str(e))   


    def _build_vector_store_(self, 
                             chunked_docs :List[Document], 
                             embedding_model : Optional[EmbeddingModel]=None) -> None:
        """
        Builds a vector store from the chunked documents using the specified embedding model.
        """
        _internal_embedding_model = True if embedding_model is None else False

        # get an embedding method with a certain model
        embed = Embeddings(internal_embedding_model=_internal_embedding_model,
                           embedding_model= embedding_model)
        #embedding_documents = embed.generate_embedding_documents(chunked_docs)
        
        
        if self.vector_db_type == "weaviate":
            self._build_vector_with_weaviate(documents=chunked_docs, 
                                             embedded_model=embed.get_auto_decided_embedding_model())
        else:
            raise ValueError(f"Unsupported vector_db_type: {self.vector_db_type}")
        

    def _build_vector_with_weaviate(self,
                                    documents : Union[List[dict], dict],
                                    embedded_model,
                                    embedding_documents :List[Document] = None):
        """
        Builds a vector store in Weaviate from the embedding documents.
        """
        weaviate_db = WeaviateDB(collection_name=self.vector_db_collection_name,
                                 embedded_model=embedded_model)
        weaviate_db.update_query(text_documents=documents)
    

    