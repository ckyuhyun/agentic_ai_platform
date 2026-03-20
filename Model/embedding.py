from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import Union, List
from enum import Enum
import hashlib

# HuggingFace Embeddings (LangChain-compatible, works with SemanticChunker)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Sentence Transformers (direct model access, fast inference)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# LlamaIndex HuggingFace (alternative integration)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False


class SearchMode(str, Enum):
    """Search modes with different accuracy/speed tradeoffs"""
    FAST = "fast"           # Quick vector search, lower accuracy
    BALANCED = "balanced"   # Default balanced search
    ACCURATE = "accurate"   # Slower but more accurate
    HYBRID = "hybrid"       # Vector + keyword fusion
    RERANK = "rerank"       # Vector search + cross-encoder reranking

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


# =============================================================================
# EMBEDDING MODEL DIMENSIONS
# =============================================================================
EMBEDDING_DIMENSIONS = {
    EmbeddingModel.MINI_LM: 384,
    EmbeddingModel.MPNET: 768,
    EmbeddingModel.BGE_SMALL: 384,
    EmbeddingModel.BGE_BASE: 768,
    EmbeddingModel.BGE_LARGE: 1024,
    EmbeddingModel.E5_SMALL: 384,
    EmbeddingModel.E5_BASE: 768,
    EmbeddingModel.E5_LARGE: 1024,
    EmbeddingModel.HASH: 384,
    EmbeddingModel.OPENAI: 1536,
}    


class Embeddings:
    def __init__(self, 
                 embedding_model: EmbeddingModel = EmbeddingModel.MINI_LM):
        self.embedding_model_name = embedding_model
        # 2. Configure the Semantic Chunker
        # 'percentile' means it splits when a gap in meaning is in the top 5% of all gaps.
        self.embeddings = None
        self.chunker = None
        
        
        self._init_embedding_model(embedding_model)
        self._set_chunker_()
        


    def _init_embedding_model(self, model_name: EmbeddingModel):
        """
        Initialize the embedding model.

        Priority order:
        1. OpenAI (if OPENAI model selected)
        2. HuggingFaceEmbeddings (LangChain-compatible, best for SemanticChunker)
        3. SentenceTransformers (direct, fast inference)
        4. LlamaIndex HuggingFace (alternative)
        5. Hash fallback
        """
        if model_name == EmbeddingModel.HASH:
            self.embeddings = None
            self._embedding_method = "hash"
            print("⚠ Using hash-based embeddings (install langchain-huggingface for better accuracy)")
            return

        # OpenAI embeddings
        if model_name == EmbeddingModel.OPENAI:
            self.embeddings = OpenAIEmbeddings(model=model_name.value)
            self._embedding_method = "openai"
            print(f"✓ Using OpenAI Embeddings: {model_name.value}")
            return

        # HuggingFace via LangChain (preferred - compatible with SemanticChunker)
        if HUGGINGFACE_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name.value,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
                )
                self._embedding_method = "huggingface"
                print(f"✓ Loaded HuggingFace embedding: {model_name.value}")
                return
            except Exception as e:
                print(f"⚠ Failed to load HuggingFace {model_name.value}: {e}")

        # SentenceTransformers (direct)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings = SentenceTransformer(model_name.value)
                self._embedding_method = "sentence_transformers"
                print(f"✓ Loaded SentenceTransformer: {model_name.value}")
                return
            except Exception as e:
                print(f"⚠ Failed to load SentenceTransformer {model_name.value}: {e}")

        # LlamaIndex HuggingFace
        if LLAMA_INDEX_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbedding(model_name=model_name.value)
                self._embedding_method = "llama_index"
                print(f"✓ Loaded LlamaIndex HuggingFace: {model_name.value}")
                return
            except Exception as e:
                print(f"⚠ Failed to load via LlamaIndex: {e}")

        # Fallback to hash
        self.embeddings = None
        self._embedding_method = "hash"
        print("⚠ Falling back to hash-based embeddings")


    def generate_embedding(self, text: str) -> Union[List[float], List[list[float]]]:
        try:
            doc = self._create_documents_([text])
        except Exception as e:
            print(f"⚠ Error creating documents for embedding: {e}")
            raise ValueError(f"{e}")

        _doc = [cont.page_content for cont in doc]
        if len(_doc) == 1:
            return self._generate_embedding(_doc[0])
        elif len(_doc) > 1:
            return self._generate_embeddings_batch(_doc)
        else:
            raise ValueError("No documents created from text for embedding generation")


    def _set_chunker_(self):
        """Initialize SemanticChunker. Requires a LangChain Embeddings instance."""
        if self._embedding_method in ("huggingface", "openai"):
            # HuggingFaceEmbeddings and OpenAIEmbeddings are LangChain-compatible
            self.chunker = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        elif self._embedding_method == "sentence_transformers" and HUGGINGFACE_AVAILABLE:
            # Wrap SentenceTransformer in LangChain-compatible HuggingFaceEmbeddings for chunker
            chunker_embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name.value,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            self.chunker = SemanticChunker(
                chunker_embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        else:
            # Fallback: no semantic chunking, return text as-is
            self.chunker = None

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for single text"""
        if self._embedding_method == "huggingface":
            return self.embeddings.embed_query(text)
        elif self._embedding_method == "openai":
            return self.embeddings.embed_query(text)
        elif self._embedding_method == "sentence_transformers":
            return self.embeddings.encode(text, convert_to_numpy=True).tolist()
        elif self._embedding_method == "llama_index":
            return self.embeddings.get_text_embedding(text)
        else:
            # Hash-based fallback
            vector_size = EMBEDDING_DIMENSIONS.get(self.embedding_model_name, 384)
            hash_value = hashlib.sha256(text.encode()).digest()
            return [((b - 128) / 128.0) for b in hash_value[:vector_size]]

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch for better performance"""
        if self._embedding_method == "huggingface":
            return self.embeddings.embed_documents(texts)
        elif self._embedding_method == "openai":
            return self.embeddings.embed_documents(texts)
        elif self._embedding_method == "sentence_transformers":
            embeddings = self.embeddings.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        elif self._embedding_method == "llama_index":
            return [self.embeddings.get_text_embedding(t) for t in texts]
        else:
            return [self._generate_embedding(t) for t in texts]


    def _create_documents_(self, texts) -> list[Union[str]]:
        """Split texts into semantic chunks. Falls back to raw texts if chunker is unavailable."""
        if self.chunker is None:
            return texts
        return self.chunker.create_documents(texts)