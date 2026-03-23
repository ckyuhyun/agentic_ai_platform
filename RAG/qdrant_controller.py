"""
Qdrant RAG Controller

A generic, high-performance vector database controller using Qdrant for semantic
search and retrieval-augmented generation (RAG) operations.

This controller is domain-agnostic and works with dynamic queries and payloads.
For domain-specific wrappers (e.g. StockPulse), see qdrant_wrapper.py.

Features:
- Multiple embedding model options (configurable)
- HNSW index optimization
- Payload indexing for fast filtering
- Hybrid search (vector + keyword)
- Query expansion hooks
- Re-ranking with cross-encoder
- Quantization for memory efficiency
- Batch operations
- Time-based cache invalidation
- gRPC connection for high throughput
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        PayloadSchemaType
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")


from agentic_ai_platform.rag.embedding import (
    EmbeddingModel,
    EMBEDDING_DIMENSIONS,
    Embeddings,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SearchMode
)
from utils.color_print import cprint


# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================

class QuantizationMode(str, Enum):
    """Quantization modes for memory optimization"""
    NONE = "none"           # No quantization (full precision)
    SCALAR_INT8 = "int8"    # Scalar quantization to INT8
    BINARY = "binary"       # Binary quantization (fastest, least accurate)


# =============================================================================
# QDRANT RAG CONTROLLER
# =============================================================================

class QdrantRAGController:
    """
    Generic Qdrant-based RAG Controller with configurable accuracy and performance tuning.

    All search/upsert operations work with dynamic query strings and arbitrary payloads.
    Domain-specific logic (ticker, article_titles, etc.) belongs in wrapper classes.

    Provides:
    - Multiple embedding model options
    - Configurable HNSW index parameters
    - Multiple search modes (fast, balanced, accurate, hybrid, rerank)
    - Quantization options for memory efficiency
    - Batch operations for throughput
    - Time-based cache invalidation
    - Pluggable query expansion via callback
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        collection_name: str = "qdrant_cache",
        embedding_model: EmbeddingModel = EmbeddingModel.MINI_LM,
        search_mode: SearchMode = SearchMode.BALANCED,
        quantization_mode: QuantizationMode = QuantizationMode.NONE,
        similarity_threshold: float = 0.85,
        use_memory: bool = False,
        path: Optional[str] = None,
        use_grpc: bool = False,
        # HNSW tuning parameters
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        hnsw_ef_search: int = 128,
        # Reranking parameters
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_k: int = 10,
        # Cache invalidation
        max_cache_age_hours: Optional[int] = None,
        # Payload indexes
        payload_indexes: Optional[List[Tuple[str, str]]] = None,
        # Query expansion hook
        query_expander: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize Qdrant RAG Controller with tuning parameters

        Args:
            host: Qdrant server host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port (for high throughput)
            collection_name: Name of the collection
            embedding_model: Which embedding model to use
            search_mode: Search mode (fast, balanced, accurate, hybrid, rerank)
            quantization_mode: Quantization for memory optimization
            similarity_threshold: Minimum similarity for cache hits (0-1)
            use_memory: Use in-memory storage (for testing)
            path: Path for local persistent storage
            use_grpc: Use gRPC for better performance
            hnsw_m: HNSW M parameter (edges per node, 16-64)
            hnsw_ef_construct: HNSW ef_construct (100-500)
            hnsw_ef_search: HNSW ef for search (higher = more accurate)
            rerank_model: Cross-encoder model for reranking
            rerank_top_k: Number of candidates for reranking
            max_cache_age_hours: Max age for cached results (None = no limit)
            payload_indexes: List of (field_name, schema_type) tuples for indexing.
                             schema_type can be "keyword", "integer", "float", "datetime".
            query_expander: Optional callback fn(query) -> expanded_query for hybrid search
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")

        self.collection_name = collection_name
        self.search_mode = search_mode
        self.quantization_mode = quantization_mode
        self.similarity_threshold = similarity_threshold
        self.max_cache_age_hours = max_cache_age_hours
        self.query_expander = query_expander

        # HNSW parameters
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.hnsw_ef_search = hnsw_ef_search

        # Reranking parameters
        self.rerank_model_name = rerank_model
        self.rerank_top_k = rerank_top_k
        self.reranker = None
        self.embedding_instance = Embeddings(embedding_model)

        # Get vector size for embedding model
        self.vector_size = EMBEDDING_DIMENSIONS.get(embedding_model, 384)

        # Payload index definitions
        self._payload_indexes = payload_indexes or [
            ("timestamp", "datetime"),
        ]

        # Initialize Qdrant client
        self._init_client(host, port, grpc_port, use_memory, path, use_grpc)

        # Initialize reranker if needed
        if search_mode == SearchMode.RERANK:
            self._init_reranker()

        # Create collection with optimized settings
        self._ensure_collection()

        # Create payload indexes for fast filtering
        self._create_payload_indexes()

    def _init_client(
        self,
        host: str,
        port: int,
        grpc_port: int,
        use_memory: bool,
        path: Optional[str],
        use_grpc: bool
    ):
        """Initialize Qdrant client with appropriate connection mode"""
        if use_memory:
            self.client = QdrantClient(":memory:")
            self._connection_mode = "memory"
        elif path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
            self._connection_mode = "local"
        elif use_grpc:
            self.client = QdrantClient(
                host=host,
                port=grpc_port,
                prefer_grpc=True,
                grpc_options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ]
            )
            self._connection_mode = "grpc"
        else:
            self.client = QdrantClient(host=host, port=port)
            self._connection_mode = "http"

        cprint(f"✓ Qdrant client initialized ({self._connection_mode} mode)")

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            from sentence_transformers import CrossEncoder
            try:
                self.reranker = CrossEncoder(self.rerank_model_name)
                print(f"✓ Loaded reranker: {self.rerank_model_name}")
            except Exception as e:
                print(f"⚠ Failed to load reranker: {e}")
                self.reranker = None
        else:
            print("⚠ Reranking requires sentence-transformers")
            self.reranker = None

    def _ensure_collection(self):
        """Create collection with optimized settings if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            return

        # Build quantization config
        quantization_config = None
        if self.quantization_mode == QuantizationMode.SCALAR_INT8:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )

        # Create collection with HNSW optimization
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            ),
            # hnsw_config : trading off between recall, latency, and memory
            hnsw_config=HnswConfigDiff(
                m=self.hnsw_m, # edges per node in the graph, Higher values improves racall but increase memory usage
                ef_construct=self.hnsw_ef_construct, # candidate list size during index build. Higher values improve racall but increase build time
                full_scan_threshold=10000 # size threahold below which Qdrant prefers brute-force scan over HNSW which can be faster for small collections
            ),
            # optimizers_config : restructure data at the segment level to keep search efficient
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000 # maximum vector size for a plain index 
            ),
            quantization_config=quantization_config
        )

        print(f"✓ Created collection: {self.collection_name}")
        print(f"  - Vector size: {self.vector_size}")
        print(f"  - HNSW M: {self.hnsw_m}, ef_construct: {self.hnsw_ef_construct}")
        print(f"  - Quantization: {self.quantization_mode.value}")

    def _create_payload_indexes(self):
        """Create payload indexes for faster filtering based on configured index definitions"""
        schema_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "datetime": PayloadSchemaType.DATETIME,
        }

        for field_name, schema_type in self._payload_indexes:
            field_schema = schema_map.get(schema_type, PayloadSchemaType.KEYWORD)
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception:
                pass  # Index may already exist

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text"""
        return hashlib.md5(text.encode()).hexdigest()

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def upsert(
        self,
        query: str,
        payload: Dict[str, Any],
        data: Any
    ) -> bool:
        """
        Insert or update a cached result

        Args:
            query: Query string used for embedding generation and ID
            payload: Dictionary containing metadata (ticker, result_type, etc.) 
            data: Result data to cache

        Returns:
            True if successful, False otherwise
        """
        if not query:
            return False

        # Generate embedding from query
        embeddings : Union[List[float], List[list[float]]] = self.embedding_instance.generate_embedding(data)

        # Generate unique ID from query
        point_id = self._generate_id(query)

        # Serialize data
        # if hasattr(data, 'model_dump'):
        #     result_data = data.model_dump()
        # elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'model_dump'):
        #     result_data = [item.model_dump() for item in data]
        # else:
        #     result_data = data

        # Prepare final payload
        final_payload = {
            "timestamp": datetime.now().isoformat(),
            #"result_data": json.dumps(result_data),
            #"query": query
        }

        # Merge with provided payload
        final_payload.update(payload)

        input_type =  None
        try:
            points : List[PointStruct] = []
            
            for embedding in embeddings:
                string_uuid = str(uuid.uuid4())
                input_type = type(embeddings)
                if not isinstance(embedding, list):
                    embedding = [embedding]

                points.append(PointStruct(
                    id=  uuid.UUID(string_uuid),
                    vector=embedding,
                    payload=final_payload
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✓ Qdrant: Upserted {len(points)} point(s)")
            return True
        except Exception as e:
            print(f"✗ Qdrant upsert error: {e} {input_type}")
            return False

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    def search(
        self,
        query: str,
        must_conditions: Optional[List[Dict[str, Union[str, int, float]]]] = None,
        deserializer: Optional[Callable] = None,
        limit: int = 1,
        search_mode: Optional[SearchMode] = None
    ) -> Optional[Any]:
        """
        Search for similar cached results using configured search mode

        Args:
            query: Query string to generate embedding and search
            must_conditions: Optional filter conditions [{"field": "value"}, ...]
            deserializer: Optional function to deserialize result data
            limit: Maximum number of results
            search_mode: Override default search mode

        Returns:
            Cached result if found with sufficient similarity, None otherwise
        """
        if not query:
            return None

        mode = search_mode or self.search_mode
        conditions = must_conditions or []

        if mode == SearchMode.FAST:
            return self._search_fast(query, conditions, deserializer, limit)
        elif mode == SearchMode.BALANCED:
            return self._search_balanced(query, conditions, deserializer, limit)
        elif mode == SearchMode.ACCURATE:
            return self._search_accurate(query, conditions, deserializer, limit)
        elif mode == SearchMode.HYBRID:
            return self._search_hybrid(query, conditions, deserializer, limit)
        elif mode == SearchMode.RERANK:
            return self._search_rerank(query, conditions, deserializer, limit)
        else:
            return self._search_balanced(query, conditions, deserializer, limit)

    def _search_fast(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable] = None,
        limit: int = 1
    ) -> Optional[Any]:
        """Fast search with minimal accuracy (uses lower ef)"""
        return self._execute_search(
            query, must_conditions, deserializer, limit,
            search_params=SearchParams(exact=False, hnsw_ef=64)
        )

    def _search_balanced(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable] = None,
        limit: int = 1
    ) -> Optional[Any]:
        """Balanced search (default settings)"""
        return self._execute_search(
            query, must_conditions, deserializer, limit,
            search_params=SearchParams(exact=False, hnsw_ef=self.hnsw_ef_search)
        )

    def _search_accurate(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable] = None,
        limit: int = 1
    ) -> Optional[Any]:
        """Accurate search (uses exact brute-force search)"""
        return self._execute_search(
            query, must_conditions, deserializer, limit,
            search_params=SearchParams(exact=True)
        )

    def _search_hybrid(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable] = None,
        limit: int = 1
    ) -> Optional[Any]:
        """Hybrid search with optional query expansion via callback"""
        # Apply query expansion if a query_expander callback is set
        expanded_query = self.query_expander(query) if self.query_expander else query

        # Generate embedding from expanded query
        query_embedding = self.embedding_instance.generate_embedding(expanded_query)

        # Build filters
        filters = self._build_filters(must_conditions)

        # Add time filter if configured
        if self.max_cache_age_hours:
            filters = self._add_time_filter(filters)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filters,
                limit=limit,
                with_payload=True,
                search_params=SearchParams(exact=False, hnsw_ef=self.hnsw_ef_search)
            )

            return self._process_search_result(response.points, deserializer)

        except Exception as e:
            print(f"✗ Qdrant hybrid search error: {e}")
            return None

    def _search_rerank(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable] = None,
        limit: int = 1
    ) -> Optional[Any]:
        """Search with cross-encoder reranking for best accuracy"""
        if not self.reranker:
            return self._search_balanced(query, must_conditions, deserializer, limit)

        query_embedding = self.embedding_instance.generate_embedding(query)

        # Build filters
        filters = self._build_filters(must_conditions)

        try:
            # First pass: get more candidates
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filters,
                limit=self.rerank_top_k,
                with_payload=True,
                search_params=SearchParams(exact=False, hnsw_ef=self.hnsw_ef_search)
            )
            candidates = response.points

            if not candidates:
                return None

            # Second pass: rerank with cross-encoder
            pairs = [(query, c.payload.get('query', '')) for c in candidates]
            scores = self.reranker.predict(pairs)

            # Sort by reranker scores
            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

            # Get best result
            best_match, rerank_score = reranked[0]

            # Normalize cross-encoder score (can be negative) via sigmoid
            normalized_score = 1 / (1 + abs(rerank_score)) if rerank_score < 0 else rerank_score

            if normalized_score < self.similarity_threshold:
                print(f"Qdrant: Rerank miss (score {normalized_score:.3f} < {self.similarity_threshold})")
                return None

            print(f"✓ Qdrant: Rerank hit - score: {rerank_score:.3f} (normalized: {normalized_score:.3f})")

            # Extract result data
            result_data = json.loads(best_match.payload['result_data'])

            if deserializer:
                return deserializer(result_data)

            return result_data

        except Exception as e:
            print(f"✗ Qdrant rerank search error: {e}")
            return None

    def _execute_search(
        self,
        query: str,
        must_conditions: List[Dict[str, Union[str, int, float]]],
        deserializer: Optional[Callable],
        limit: int,
        search_params: Optional[SearchParams] = None,
    ) -> Optional[Any]:
        """Execute search with given parameters"""
        query_embedding = self.embedding_instance.generate_embedding(query)
 
        # Build filters
        filters = self._build_filters(must_conditions)

        # Add time filter if configured
        if self.max_cache_age_hours:
            filters = self._add_time_filter(filters)

        if search_params is None:
            search_params = SearchParams(exact=False, hnsw_ef=self.hnsw_ef_search)      

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filters,
                limit=limit,
                with_payload=True,
                search_params=search_params
            )

            return self._process_search_result(response.points, deserializer)

        except Exception as e:
            print(f"✗ Qdrant search error: {e}")
            return None

    def _build_filters(self, must_conditions: List[Dict[str, Union[str, int, float]]]) -> Filter:
        """Build Qdrant filter from conditions"""
        must_filters = []
        for condition in must_conditions:
            for key, value in condition.items():
                must_filters.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=must_filters)

    def _add_time_filter(self, filters: Filter) -> Filter:
        """Add time-based filter for cache freshness"""
        cutoff_time = (datetime.now() - timedelta(hours=self.max_cache_age_hours)).isoformat()

        time_condition = FieldCondition(
            key="timestamp",
            range=models.Range(gte=cutoff_time)
        )

        if filters.must:
            filters.must.append(time_condition)
        else:
            filters.must = [time_condition]

        return filters

    def _process_search_result(
        self,
        response: List,
        deserializer: Optional[Callable]
    ) -> Optional[Any]:
        """Process search response and return result"""
        if not response:
            return None

        best_match = response[0]
        similarity = best_match.score

        # Check similarity threshold
        if similarity < self.similarity_threshold:
            print(f"Qdrant: Cache miss (similarity {similarity:.3f} < threshold {self.similarity_threshold})")
            return None

        print(f"✓ Qdrant: Cache hit - similarity: {similarity:.3f}")

        # Extract result data
        try:
            result_data = json.loads(best_match.payload['article_titles'])
        except Exception as e:
            cprint(f"Error deserializing search result: {e}", color="red")
            return None

        if deserializer:
            return deserializer(result_data)

        return result_data

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def batch_upsert(
        self,
        items: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert or update multiple cached results

        Args:
            items: List of dicts with keys: query, payload, data

        Returns:
            Number of successfully upserted items
        """
        if not items:
            return 0

        # Generate all embeddings in batch
        queries = [item['query'] for item in items]
        embeddings = self.embedding_instance._generate_embeddings_batch(queries)

        points = []
        for item, embedding in zip(items, embeddings):
            query = item['query']
            payload = item.get('payload', {})
            data = item['data']

            # Serialize data
            if hasattr(data, 'model_dump'):
                result_data = data.model_dump()
            elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'model_dump'):
                result_data = [d.model_dump() for d in data]
            else:
                result_data = data

            # Prepare payload
            final_payload = {
                "timestamp": datetime.now().isoformat(),
                "result_data": json.dumps(result_data),
                "query": query
            }
            final_payload.update(payload)

            points.append(
                PointStruct(
                    id=self._generate_id(query),
                    vector=embedding,
                    payload=final_payload
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✓ Qdrant: Batch upserted {len(points)} items")
            return len(points)
        except Exception as e:
            print(f"✗ Qdrant batch upsert error: {e}")
            return 0

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    def delete(
        self,
        must_conditions: Optional[List[Dict[str, Union[str, int, float]]]] = None,
    ) -> bool:
        """
        Delete cached entries matching filter conditions.
        If no conditions are provided, clears the entire collection.

        Args:
            must_conditions: Filter conditions [{"field": "value"}, ...].
                             If empty list or None, clears entire collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            if must_conditions:
                filters = self._build_filters(must_conditions)
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=filters)
                )
                print(f"✓ Qdrant: Deleted cache with {len(must_conditions)} condition(s)")
            else:
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
                self._create_payload_indexes()
                print("✓ Qdrant: Cleared entire collection")

            return True
        except Exception as e:
            print(f"✗ Qdrant delete error: {e}")
            return False

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics and configuration"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_points": info.points_count,
                "vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
                "vector_size": self.vector_size,
                "search_mode": self.search_mode.value,
                "quantization": self.quantization_mode.value,
                "similarity_threshold": self.similarity_threshold,
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_construct": self.hnsw_ef_construct,
                "hnsw_ef_search": self.hnsw_ef_search,
                "connection_mode": self._connection_mode,
                "max_cache_age_hours": self.max_cache_age_hours
            }
        except Exception as e:
            return {"error": str(e)}

    def scroll(self) -> List[Any]:
        """Scroll through all points in the collection (for debugging) and returns"""
        try:
            points = []
            offset = None

            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                points.extend(response)
                #offset = response.next_page_offset

                if offset is None:
                    break

            return points
        except Exception as e:
            print(f"Error scrolling collection: {e}")
            return []
        
    def scroll_by_field(
        self,
        field_name: str,
        batch_size: int = 100
    ) -> List[Any]:
        """Scroll through all points and collect unique values of a payload field"""
        try:
            values = set()
            offset = None

            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=[field_name]
                )

                for point in results:
                    if point.payload and field_name in point.payload:
                        values.add(point.payload[field_name])

                if offset is None:
                    break

            return sorted(list(values))
        except Exception as e:
            print(f"Error scrolling field '{field_name}': {e}")
            return []

    def set_search_mode(self, mode: SearchMode):
        """Change search mode at runtime"""
        self.search_mode = mode
        if mode == SearchMode.RERANK and not self.reranker:
            self._init_reranker()
        print(f"✓ Search mode changed to: {mode.value}")

    def set_similarity_threshold(self, threshold: float):
        """Change similarity threshold at runtime"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        print(f"✓ Similarity threshold changed to: {self.similarity_threshold}")

    def set_query_expander(self, expander: Optional[Callable[[str], str]]):
        """Set or clear the query expansion callback for hybrid search"""
        self.query_expander = expander

