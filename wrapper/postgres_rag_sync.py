from agentic_ai_platform.data.weaviate_property_data import WeaviateProperty
from agentic_ai_platform.db.postgres import PostgresDB
from agentic_ai_platform import weaviate

class DBSync:
    def __init__(self):
        self.postgresDB = PostgresDB(host='localhost',
                                port=5433)
        
        

    def sync_postgres_to_weaviate(self):
        fetched_data = self.postgresDB \
                           .fetch_all(query=
                                  """
                                    SELECT c.id, c.document_id, c.content, d.category 
                                                    FROM document_chunks c
                                                    JOIN enterprise_documents d ON c.document_id = d.id
                                                    WHERE c.sync_status = 'PENDING';
                                  """)
        
        if not fetched_data:
            return 
        
        wvc_property :list[WeaviateProperty] = []

        for d in fetched_data:
            chunk_id, doc_id , content, category  = d
            wvc_property.append(WeaviateProperty(
                postgres_chunk_id=str(chunk_id),
                document_id=str(doc_id),
                content = content,
                category=category
            ))


        
        # update data
        weaviate.properties_config = wvc_property
        weaviate.update_query(text_documents=fetched_data)


        

        
        
            


