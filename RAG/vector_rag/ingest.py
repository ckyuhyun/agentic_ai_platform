import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    WordDocumentLoader, 
    DirectoryLoader)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



from typing import List, Literal, Optional, Union

from agentic_ai_platform.db.weaviate_db import WeaviateDB
from agentic_ai_platform.graph.embedded_model_decision import EmbeddedModelDecision
from agentic_ai_platform.rag.embedding import Embeddings
from agentic_ai_platform.rag.embedded_model_list import EmbeddingModel


class Ingest:

    def __init__(self,
                 vector_db_type : Literal["weaviate"],
                 vector_db_collection_name: str):
        self .vector_db_type = vector_db_type
        self.vector_db_collection_name = vector_db_collection_name



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


    def _load_documents_from_directory_(directory_path:str, 
                                    file_type: Literal["pdf", "txt", "docx"]) -> List[Document]:
        """
        Loads documents from a specified directory based on the file type.
        """
        loaders = DirectoryLoader(path=directory_path, 
                                glob=f"**/*.{file_type}", 
                                show_progress=True)
        
        docs = []

        for loader in loaders.load():
            docs.append(loader)
        

        return docs


    def _get_chunk_documents_(self,
                              documents :List[Document], 
                              chunk_size : int =20, 
                              chunk_overlap : int=5,
                              char_offset_add : bool =True) -> List[Document]:
        """
        Chunks the documents into smaller pieces using RecursiveCharacterTextSplitter.
        """    

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len)

        return text_splitter.split_documents(documents=documents)


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
        embedding_documents = embed.generate_embedding_documents(chunked_docs)
        
        
        if self.vector_db_type == "weaviate":
            self._build_vector_with_weaviate(embedding_documents)
        else:
            raise ValueError(f"Unsupported vector_db_type: {self.vector_db_type}")
        

    def _build_vector_with_weaviate(self,
                                    embedding_documents :List[Document]):
        """
        Builds a vector store in Weaviate from the embedding documents.
        """
        weaviate_db = WeaviateDB(collection_name=self.vector_db_collection_name)
        weaviate_db.update_query(embedding_documents)
    

    