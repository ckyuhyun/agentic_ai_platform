import os

from agentic_ai_platform.rag.embedding import EmbeddingModel

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


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


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class EmbeddedModelDecision:
    def __init__(self,
                 text: str):
        self.embeddings = None
        self.text = text
        self.model = None
        self._embedding_method = None
        self.__oallama_embedding_mode__()
    
    def get_auto_decided_embedding_model(self):
         return self.embeddings
    
    def get_embedding_method(self):
        return self._embedding_method

        
    def __text_length__(self):
        return len(self.text)
    
    def __oallama_embedding_mode__(self):      
           method = "nomic-embed-text"

           self.embeddings = OllamaEmbeddings(
                model=method,
                base_url=OLLAMA_BASE_URL
            )
           self._embedding_method= method


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