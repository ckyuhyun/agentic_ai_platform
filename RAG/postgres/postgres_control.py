import asyncio
import os
from typing import Any, overload

from llama_index.core.query_engine import RetrieverQueryEngine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import inspect

from agentic_ai_platform import Postgres_Engine
from agentic_ai_platform.embedding.embedding_service import EmbeddingService
from agentic_ai_platform.RAG.postgres.sqlModel_retriever import SqlModelRetriever
from agentic_ai_platform.llm.llama_index_llm import LlamaIndexLLM




class PostgresController:
    """
    This is a main instance functioning a push, a search and so on over postgres.
    each sub class per certain model would be implemented with this class as a parent class.
    
    """
    def __init__(self):        
        """
        model : ORM data model
        table_name : table name
        content_field_name : field name 
        vecto__model__ld_name : embedded field name being searchable
        """
        self.__model__ : Any
        self.__table_name__ : str
        self.__content_field_name__:str
        self.__vector_field_name__:str

        self.__embeddings__ = EmbeddingService()
        self.__response_llm__ = LlamaIndexLLM(model_name=os.getenv("LLM_Model_Key"))
        
        self.__Base_ = declarative_base()
        self._session_factory = self.__session_create__()
        
         
    
    
    async def push(self, 
                   stmt:Any):
        await asyncio.to_thread(self.__push__, stmt=stmt)

  

    async def fetch(self,
                    stmt:Any) -> Any:
        return await asyncio.to_thread(self.__fetch__, stmt)


    
    async def search_query(self,
                           query:str):

        if not self.__model__:
            raise 'The llm model for postgres was not set from inherited child class '
        
        sm = SqlModelRetriever(session_factory=self.__session_create__,
                                target_model=self.__model__,
                                content_field_name=self.__content_field_name__,
                                vector_field_name=self.__vector_field_name__)
        
        query_engine = RetrieverQueryEngine.from_args(
                    retriever=sm,
                    llm=self.__response_llm__,
                    #node_postprocessors=[CohereRerank(top_n=3)]
                )
        
        return await query_engine.aquery(query)

    
    def init_table(self):
        if not self.__table_name__:
            raise ValueError("table name is not set")

        if not self.__model__:
            raise 'The llm model for postgres was not set from inherited child class '
            
        inspector = inspect(Postgres_Engine)
    
        if not inspector.has_table(self.__table_name__):
            # Create all tables in the engine
            self.__model__.metadata.create_all(Postgres_Engine)

    def __fetch__(self, 
                  stmt:Any) -> Any:
        try:
            with self.__session_create__(expire_on_commit=False) as session:
                self._session_factory
                result = session.execute(stmt)
                return result.mappings().all()
        except:
            raise            

    def __push__(self, 
                 stmt:Any):
        with self.__session_create__() as session, session.begin():
                        session.execute(stmt)            

    
    def __session_create__(self, 
                           expire_on_commit=True):
        self._session_factory = sessionmaker(bind=Postgres_Engine, 
                     expire_on_commit=expire_on_commit)
        return self._session_factory()

    

         
        

        
        
        
        

        


        



    
