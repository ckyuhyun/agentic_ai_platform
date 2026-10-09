from enum import Enum

class EmbeddingModel(str, Enum):
    """Available embedding models with different accuracy/speed tradeoffs"""
    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384 dims
    MPNET = "sentence-transformers/all-mpnet-base-v2"   # Balanced, 768 dims
    BGE_SMALL = "BAAI/bge-small-en-v1.5"                # Good accuracy, 384 dims
    BGE_BASE = "BAAI/bge-base-en-v1.5"                  # Better accuracy, 768 dims
    BGE_LARGE = "BAAI/bge-large-en-v1.5"                # Best accuracy, 1024 dims
    E5_SMALL = "intfloat/e5-small-v2"                   # Fast, 384 dims
    E5_BASE = "intfloat/e5-base-v2"                     # Balanced, 768 dims
    E5_LARGE = "intfloat/e5-large-v2"                   # Best, 1024 dims
    HASH = "hash"                                       # Fallback hash-based
    OPENAI = "text-embedding-3-small"                   # OpenAI Embeddings, 1536 dims