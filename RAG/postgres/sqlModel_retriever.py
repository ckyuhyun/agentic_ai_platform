import os
import numpy as np
from typing import List, Any

from sqlmodel import SQLModel, Session, select
from sqlalchemy import Column, inspect
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import QueryBundle, Settings

from agentic_ai_platform.embedding.embedding_service import EmbeddingService
from agentic_ai_platform.embedding.llamaindex_embedding import LLamaIndexEmbeddings

class SqlModelRetriever(BaseRetriever):
    def __init__(self,
                 session_factory:Any,
                 target_model:type[SQLModel],
                 vector_field_name:str="embedding", # Name of the pgvector field
                 content_field_name:str="content",  # Field to use for LlamaIndex Node text
                 join_model: type[SQLModel]|None = None,
                 join_condition : Any|None=None,
                 where_clauses:List[Any]|None=None,
                 min_score:float = 0.70,
                 top_k:int=5):
        
        self.session_factory = session_factory
        self.target_model = target_model
        self.vector_field = getattr(target_model, vector_field_name)
        self.content_field_name = content_field_name
        
        self.join_model = join_model
        self.join_condition = join_condition
        self.where_clauses = where_clauses or []
        self.min_score = min_score
        self.top_k = top_k
        Settings.embed_model = LLamaIndexEmbeddings(name=os.getenv("TEI_EMBEDDING_MODEL_NAME"),
                                                    bases=EmbeddingService)
        super().__init__()
        

    def _retrieve(self, 
                  query_bundle) -> List[NodeWithScore]:
        # 1. Automatic Query Vectorization via LlamaIndex Config
        query_vector = Settings.embed_model.get_query_embedding(query_bundle.query_str)
        return self._search(query_vector)

    async def _aretrieve(self, query_bundle) -> List[NodeWithScore]:
        query_vector = await Settings.embed_model.aget_query_embedding(query_bundle.query_str)

        return self._search(query_vector)
    

    def _search(self, query_vector):
        embedding_arr = np.array(query_vector)
        
        # 2. Dynamic Cosine Distance Formula using reflection
        similarity_expr = (1.0 - self.vector_field.cosine_distance(embedding_arr)).label("similarity_score")

        # 3. Dynamic Selection Construction
        stmt = select(self.target_model, similarity_expr)

        # Handle Dynamic SQL Join (If passed)
        if self.join_model and self.join_condition is not None:
            stmt = stmt.join(self.join_model, self.join_condition)
        
        # Apply Dynamic Where Clauses
        for claude  in self.where_clauses:
            stmt = stmt.where(claude)

        # Filter by threshold and order by best match
        stmt = stmt.where(similarity_expr > self.min_score).order_by(similarity_expr.desc()).limit(self.top_k)


        nodes=[]
        with self.session_factory() as session:
            results =  session.execute(stmt).all()

            for row in results:
                # Disambiguate when query returns (Model, similarity_score)
                record = row[0]
                score = float(row[1])

                # Dynamically extract primary key ID and content text
                primary_key = inspect(record).identity[0]
                content_text  = getattr(record, self.content_field_name, "")

                # Extract remaining fields automatically into metadata
                # metadata = {
                #     k:v for k,v in record.model_dump().items()
                #     if k not in [self.content_field_name, "embedding"]
                # }

                node = TextNode(
                    text=content_text,
                    id_=str(primary_key),
                    #metadata=metadata
                )
                nodes.append(NodeWithScore(node=node, score=score))

        return nodes


                 


        


    
