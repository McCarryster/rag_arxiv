from typing import List, Dict, Any, Optional, Union, cast
import hashlib
import json
import redis
import numpy as np

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
        self.CORPUS_VERSION_KEY: str = "corpus_version"
        
        self._setup_semantic_index()

    def get_corpus_version(self) -> int:
        value: Optional[str] = cast(Optional[str], self.client.get(self.CORPUS_VERSION_KEY))
        return int(value) if value is not None else 0

    def bump_corpus_version(self) -> int:
        value: int = cast(int, self.client.incr(self.CORPUS_VERSION_KEY))
        return value

    def _setup_semantic_index(self) -> None:
        try:
            self.client.ft(self.index_name).info()
        except Exception:
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
                    },
                ),
            ]
            definition: IndexDefinition = IndexDefinition(
                prefix=[self.SEMANTIC_PREFIX],
                index_type=IndexType.HASH,
            )
            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )

    def _get_exact_hashkey(self, query_text: str, corpus_version: int) -> str:
        clean_text: str = query_text.strip().lower()
        hash_val: str = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        return f"{self.EXACT_PREFIX}{corpus_version}:{hash_val}"

    # ------------------------------------------------------------------
    # Exact cache
    # ------------------------------------------------------------------
    def get_exact_cache(
        self,
        query_text: str,
        corpus_version: int,
    ) -> Optional[Dict[str, Any]]:
        key: str = self._get_exact_hashkey(query_text, corpus_version)
        cached: Optional[str] = cast(Optional[str], self.client.get(key))
        return json.loads(cached) if cached else None

    def set_exact_cache(
        self,
        query_text: str,
        data: Dict[str, Any],
        corpus_version: int,
        ttl: int = 3600,
    ) -> None:
        key: str = self._get_exact_hashkey(query_text, corpus_version)
        self.client.setex(key, ttl, json.dumps(data))

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------
    def get_semantic_cache(
        self,
        query_vector: List[float],
        corpus_version: int,
        threshold: float = 0.15,
    ) -> Optional[Dict[str, Any]]:
        vector_blob: bytes = np.array(query_vector, dtype=np.float32).tobytes()

        query: Query = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score", "json_data")
            .dialect(2)
        )

        params: Dict[str, Union[str, int, float, bytes]] = {"vec": vector_blob}

        try:
            results: Any = self.client.ft(self.index_name).search(
                query,
                query_params=params,
            )

            if not results.docs:
                return None

            doc: Any = results.docs[0]
            score: float = float(doc.score)

            if score > threshold:
                return None

            data: Dict[str, Any] = json.loads(doc.json_data)

            # Corpus version mismatch -> invalidate
            if data.get("corpus_version") != corpus_version:
                return None

            # Never reuse semantic cache without sources
            if not data.get("sources"):
                return None

            return {
                "score": score,
                "data": data,
            }

        except Exception as exc:
            print(f"Semantic Cache Search Error: {exc}")

        return None

    def set_semantic_cache(
        self,
        query_text: str,
        query_vector: List[float],
        data: Dict[str, Any],
        corpus_version: int,
        ttl: int = 86400,
    ) -> None:
        # Do NOT store semantic cache if retrieval was empty
        if not data.get("sources"):
            return

        clean_text: str = query_text.strip().lower()
        hash_val: str = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        key: str = f"{self.SEMANTIC_PREFIX}{corpus_version}:{hash_val}"

        vector_blob: bytes = np.array(query_vector, dtype=np.float32).tobytes()

        payload: Dict[str, Union[str, bytes]] = {
            "query_text": query_text,
            "embedding": vector_blob,
            "json_data": json.dumps(data),
        }

        self.client.hset(key, mapping=payload)
        self.client.expire(key, ttl)