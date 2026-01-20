from typing import Any, Dict, List
import hashlib
import uuid

import weaviate
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
)
from weaviate.classes.query import HybridFusion

import config

class WeaviateDBManager:
    def __init__(self, collection_name: str, index_name: str):
        # self._client: weaviate.WeaviateClient = weaviate.connect_to_local()
        self._client: weaviate.WeaviateClient = weaviate.connect_to_custom(
            http_host=config.WEAVIATE_HTTP_HOST,
            http_port=config.WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=config.WEAVIATE_GRPC_HOST,
            grpc_port=config.WEAVIATE_GRPC_PORT,
            grpc_secure=False,
        )

        self._collection_name: str = collection_name
        self._index_name: str = index_name
        self._ensure_collection()
        self._collection = self._client.collections.get(self._collection_name)

    def close(self) -> None:
        self._client.close()

    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        """
        if self._client.collections.exists(self._collection_name):
            return

        self._client.collections.create(
            name=self._collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="file_hash", data_type=DataType.TEXT, index_filterable=True),
                Property(name="chunk_id", data_type=DataType.INT, index_filterable=True),
                Property(name="source", data_type=DataType.TEXT),  # optional, if you need it
            ],
            vector_config=Configure.Vectors.self_provided(),
        )

    def _deterministic_uuid(self, file_hash: str, chunk_id: int) -> uuid.UUID:
        """
        Deterministic UUID for deduplication.
        """
        raw_key: str = f"{self._index_name}:{file_hash}:{chunk_id}"
        digest: bytes = hashlib.sha256(raw_key.encode("utf-8")).digest()
        return uuid.UUID(bytes=digest[:16])

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Insert vectors with deterministic UUIDs (idempotent).
        """
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts, embeddings, and metadatas length mismatch")

        collection = self._client.collections.get(self._collection_name)

        with collection.batch.dynamic() as batch:
            for i in range(len(texts)):
                embedding: List[float] = embeddings[i]
                metadata: Dict[str, Any] = metadatas[i]

                if "file_hash" not in metadata or "chunk_id" not in metadata:
                    raise ValueError("metadata must contain 'file_hash' and 'chunk_id'")

                object_id: uuid.UUID = self._deterministic_uuid(
                    file_hash=str(metadata["file_hash"]),
                    chunk_id=int(metadata["chunk_id"]),
                )

                batch.add_object(
                    uuid=object_id,
                    properties={
                        "text": texts[i],
                        "file_hash": metadata["file_hash"],
                        "chunk_id": metadata["chunk_id"],
                    },
                    vector=embedding,
                )

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + vector search for the current collection.
        """
        kwargs: Dict[str, Any] = {
            "alpha": alpha,
            "limit": limit,
            "fusion_type": HybridFusion.RELATIVE_SCORE,
            "return_properties": ["text", "file_hash", "chunk_id"],
        }
        if query_embedding is not None:
            # matches docs: vector + query + alpha + limit
            kwargs["vector"] = query_embedding

        response = self._collection.query.hybrid(
            query=query_text,
            **kwargs,
        )

        results: List[Dict[str, Any]] = []
        for obj in response.objects:
            props = obj.properties
            results.append(
                {
                    "id": obj.uuid,
                    "text": props.get("text"),
                    "file_hash": props.get("file_hash"),
                    "chunk_id": props.get("chunk_id"),
                }
            )

        return results