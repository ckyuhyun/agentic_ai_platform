
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader, 
    word_document, 
    DirectoryLoader)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



from typing import List, Literal, Optional

from agentic_ai_platform.db.weaviate_db import WeaviateDB
from agentic_ai_platform.graph.embedded_model_decision import EmbeddedModelDecision
from agentic_ai_platform.rag.embedding import Embeddings
from agentic_ai_platform.rag.embedded_model_list import EmbeddingModel


def load_vector_store(directory_path:str, 
                                  file_type: Literal["pdf", "txt", "docx"]):
    documents = _load_documents_from_directory_(directory_path, file_type)
    chunked_docs = _get_chunk_documents_(documents)
    _build_vector_store_(chunked_docs, vector_store=None, db_type="weaviate", embedding_model=None)


def _load_documents_from_directory_(directory_path:str, 
                                  file_type: Literal["pdf", "txt", "docx"]):
    """
    Loads documents from a specified directory based on the file type.
    """
    
    loaders = DirectoryLoader(path=directory_path, 
                              glob=f"**/*.{file_type}", 
                              show_progress=True)
    
    docs = []

    for loader in loaders.load():
        docs.extend(loader)
    

    return docs


def _get_chunk_documents_(documents :List[Document], 
                    chunk_size : int =100, 
                    chunk_overlap : int=200,
                    char_offset_add : bool =True) -> List[Document]:
    """
    Chunks the documents into smaller pieces using RecursiveCharacterTextSplitter.
    """    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len,
                                                   char_offset_add=char_offset_add)

    return text_splitter.split_documents(documents=documents)


def _build_vector_store_(chunked_docs :List[Document], 
                         vector_store,                          
                         db_type : Literal["weaviate"],
                         embedding_model : Optional[EmbeddingModel]=None) -> None:
    """
    Builds a vector store from the chunked documents using the specified embedding model.
    """
    _internal_embedding_model = True if embedding_model is None else False
    
    # get an instance 
    emd = EmbeddedModelDecision(internal_embedding_model=_internal_embedding_model) if _internal_embedding_model else  EmbeddedModelDecision(internal_embedding_model=_internal_embedding_model, model_name=embedding_model)
    
    # get an embedding model instance
    embedded_model = emd.get_auto_decided_embedding_model()

    # get an embedding method with a certain model
    embed = Embeddings(embedding_model)
    embedding_documents = embed.generate_embedding_documents(chunked_docs)
    
    
    if db_type == "weaviate":
        _build_vector_with_weaviate(embedding_documents)
    else:
        raise ValueError(f"Unsupported db_type: {db_type}")
    

def _build_vector_with_weaviate(embedding_documents :List[Document]):
    weaviate_db = WeaviateDB()
    weaviate_db.update_query(embedding_documents)
    

    