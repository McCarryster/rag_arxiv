import json
import hashlib
import numpy as np
import redis
from typing import List, Dict, Any, Optional, Union, cast, Mapping
from redis.commands.search.field import VectorField, TextField, Field
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class RedisCacheManager:
    def __init__(
        self, 
        RD: Dict[str, Any], 
        vector_dim: int = 1536, 
        index_name: str = "semantic_cache_idx" 
    ) -> None:
        """
        Initializes the dual-caching manager with RediSearch capabilities.
        """
        self.redis_config: Dict[str, Any] = RD
        self.client: redis.Redis = redis.Redis(**self.redis_config, decode_responses=True)
        
        self.vector_dim: int = vector_dim
        self.index_name: str = index_name
        self.EXACT_PREFIX: str = "exact:"
        self.SEMANTIC_PREFIX: str = "sem:"
        
        self._setup_semantic_index()

    def _setup_semantic_index(self) -> None:
        """Sets up the RediSearch index for vector similarity search."""
        try:
            self.client.ft(self.index_name).info()
        except Exception:
            # SCHEMA CHANGED: 'payload' renamed to 'json_data' to avoid Document() collision
            schema: List[Any] = [
                TextField("query_text"),
                TextField("json_data"), 
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                ),
            ]
            definition: IndexDefinition = IndexDefinition(
                prefix=[self.SEMANTIC_PREFIX], 
                index_type=IndexType.HASH
            )
            self.client.ft(self.index_name).create_index(fields=schema, definition=definition)

    def _get_exact_hashkey(self, query_text: str) -> str:
        """Generates a deterministic MD5 key for exact string matches."""
        clean_text: str = query_text.strip().lower()
        hash_val: str = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        return f"{self.EXACT_PREFIX}{hash_val}"

    def get_exact_cache(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Retrieves data if the query string is an exact match."""
        key: str = self._get_exact_hashkey(query_text)
        cached: Optional[str] = cast(Optional[str], self.client.get(key))
        return json.loads(cached) if cached else None

    def set_exact_cache(self, query_text: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """Stores data keyed by the exact query string."""
        key: str = self._get_exact_hashkey(query_text)
        self.client.setex(key, ttl, json.dumps(data))

    def get_semantic_cache(
        self, 
        query_vector: List[float], 
        threshold: float = 0.15
    ) -> Optional[Dict[str, Any]]:
        """
        Searches for a semantically similar query in Redis using KNN.
        """
        # Convert vector list to binary bytes
        vector_blob: bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        # LOGIC CHANGED: Request 'json_data' instead of 'payload'
        q: Query = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score", "json_data")
            .dialect(2)
        )
        
        params: Dict[str, Union[str, int, float, bytes]] = {"vec": vector_blob}
        
        try:
            results: Any = self.client.ft(self.index_name).search(q, query_params=params)
            if results.docs:
                res: Any = results.docs[0]
                score: float = float(res.score)
                
                if score <= threshold:
                    # Access the renamed field via attribute
                    return {
                        "score": score,
                        "data": json.loads(res.json_data)
                    }
        except Exception as e:
            print(f"Semantic Cache Search Error: {e}")
            
        return None

    def set_semantic_cache(
        self, 
        query_text: str, 
        query_vector: List[float], 
        data: Dict[str, Any]
    ) -> None:
        """
        Stores the query and its results in the vector index.
        """
        # Generate a unique key for the semantic hash
        clean_text: str = query_text.strip().lower()
        hash_val: str = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        key: str = f"{self.SEMANTIC_PREFIX}{hash_val}"
        
        vector_blob: bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        # LOGIC CHANGED: Store in 'json_data' instead of 'payload'
        mapping: Dict[str, Union[str, bytes]] = {
            "query_text": query_text,
            "embedding": vector_blob,
            "json_data": json.dumps(data)
        }
        
        self.client.hset(key, mapping=mapping)