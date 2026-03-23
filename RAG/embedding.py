
from typing import Union, List
from enum import Enum
import hashlib
from agentic_ai_platform.model.embedded_model_list import EmbeddingModel
from agentic_ai_platform.graph.embedded_model_decision import EmbeddedModelDecision
from langchain_experimental.text_splitter import SemanticChunker


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
        emd = EmbeddedModelDecision("")
        self.embeddings = emd.get_auto_decided_embedding_model()
        self._embedding_method = emd.get_embedding_method()
        self.chunker = None

        self._set_chunker_()
        


    


    def generate_embedding(self, text: str) -> Union[List[float], List[list[float]]]:
        try:
            doc = self._create_documents_([text])
        except Exception as e:
            print(f"⚠ Error creating documents for embedding: {e}")
            raise ValueError(f"{e}")

        _doc = [cont for cont in doc]
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
        elif self._embedding_method == "nomic-embed-text":
            # OLLAMA nomic-embed-text embedding mode (using OllamaEmbeddings interface)
            return self.embeddings.embed_query(text)
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