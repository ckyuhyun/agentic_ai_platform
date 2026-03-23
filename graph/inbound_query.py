from agentic_ai_platform.rag.embedding import Embeddings

class InboundQuery:
    def __init__(self, 
                 query: str):
        self._query_ = query
        self._embedding_ = None
        self._embedding_processing_()

    def get_embedding(self):
        return self._embedding_


    def _embedding_processing_(self):
        emd = Embeddings()
        self._embedding_ = emd.generate_embedding(self._query_)        


    