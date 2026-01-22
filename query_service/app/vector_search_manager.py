from typing import List, Dict, Any, Optional
import tiktoken
import asyncio
from fastapi.concurrency import run_in_threadpool
import httpx

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langfuse import Langfuse
from langfuse_decorator import async_trace

import config


# Manager for vector search operations
class VectorSearchManager:
    def __init__(self, embedding_model: OpenAIEmbeddings) -> None:
        self.embedding_model: OpenAIEmbeddings = embedding_model

    async def _post_to_vector_db_async(self, payload: Dict[str, Any], endpoint_url: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url=endpoint_url,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def _wrapper_with_retry(
        self,
        payload: Dict[str, Any],
        endpoint_url: str,
        retries: int = 5,
        delay_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None

        for _ in range(retries):
            try:
                return await self._post_to_vector_db_async(payload, endpoint_url)
            except Exception as exc:
                last_exception = exc
                await asyncio.sleep(delay_seconds)

        raise RuntimeError(
            f"Failed to find relevant docs in vector DB after retries: {last_exception}"
        )

    async def perform_hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int = 5,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Search vector DB using a hybrid (BM25 + vector) query.
        """
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "limit": limit,
            "alpha": alpha,
        }
        response_json = await self._wrapper_with_retry(payload, config.VECTOR_DB_HYBRID_SEARCH_URL)
        
        # Convert response to Document objects
        results: List[Document] = []
        for r in response_json.get("results", []):
            results.append(
                Document(
                    page_content=r.get("text", ""),
                    metadata={
                        "id": r.get("id"),
                        "chunk_id": r.get("chunk_id"),
                        "file_hash": r.get("file_hash"),
                    },
                )
            )
        return results

    async def get_docs_by_metadata(
        self,
        cached_metadata: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Document]:
        """
        Search vector DB by metadata filters.
        """
        payload: Dict[str, Any] = {
            "filters": cached_metadata,
            "limit": limit,
        }
        response_json = await self._wrapper_with_retry(payload, endpoint_url=config.VECTOR_DB_METADATA_SEARCH_URL)

        results: List[Document] = []
        for r in response_json.get("results", []):
            results.append(
                Document(
                    page_content=r.get("text", ""),
                    metadata={
                        "file_hash": r.get("file_hash"),
                        "chunk_id": r.get("chunk_id"),
                        "id": r.get("id"),
                    },
                )
            )
        return results

    @async_trace(name="query pipeline", model=config.TEXT_EMBEDDING_MODEL)
    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        embedding: List[float] = await run_in_threadpool(self.embedding_model.embed_query, text)
        encoding = tiktoken.encoding_for_model(config.TEXT_EMBEDDING_MODEL)
        total_input_tokens: int = len(encoding.encode(text))

        print("[DEBUG]: LANFUSE", "Shit happened", flush=True)
        
        result: Dict[str, Any] = {
            "result": embedding,
            "input": text,
            "output": len(embedding),
            "input_tokens": total_input_tokens,
            "total_tokens": total_input_tokens,
            "metadata": {
                "text_length": len(text),
                "embedding_dimensions": len(embedding)
            }
        }
        return result