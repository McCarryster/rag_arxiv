import hashlib
from fastapi.concurrency import run_in_threadpool
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
import tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langfuse import Langfuse

from redis_manager import RedisManager
import config

# Manager for vector db operations
class VectorStoreManager:
    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        duplicate_tracker: RedisManager,
        index_name: str,
        langfuse_tracing = None
    ) -> None:
        self.embedding_model: OpenAIEmbeddings = embedding_model
        self.text_splitter: RecursiveCharacterTextSplitter = text_splitter
        self.tracker: RedisManager = duplicate_tracker
        self.index_name: str = index_name
        self.langfuse_tracing = langfuse_tracing

    def get_file_hash(self, file_path: str) -> str:
            """
            Generates a SHA-256 hash of the file content.
            SHA-256 is more collision-resistant than MD5 for high-volume data.
            """
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                # 64KB chunks are efficient for modern CPUs and filesystems
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.langfuse_tracing:
            trace_summary: Dict[str, Any] = {
                "num_texts": len(texts),
                "total_character_length": sum(len(t) for t in texts),
                "avg_chunk_length": sum(len(t) for t in texts) / len(texts) if texts else 0
            }
            embedding_obs = self.langfuse_tracing.start_observation(
                name="openai_embeddings_generation",
                as_type="embedding",
                model=config.TEXT_EMBEDDING_MODEL,
                input=trace_summary
            )
            try:
                embeddings: List[List[float]] = self.embedding_model.embed_documents(texts)
                # Calculate token usage
                encoding = tiktoken.encoding_for_model(config.TEXT_EMBEDDING_MODEL)
                total_input_tokens: int = sum(len(encoding.encode(t)) for t in texts)
                # Update observation with usage and metadata
                embedding_obs.update(
                    usage={
                        "input": total_input_tokens,
                        "total": total_input_tokens,
                    },
                    output=[{"dimensions": len(e)} for e in embeddings]  # log the metadata/dimensions rather than the full vectors to save space
                )
                embedding_obs.end()
                # self.langfuse_tracing.flush()
                return embeddings
            except Exception as e:
                embedding_obs.update(level="ERROR", status_message=str(e))  # Ensure errors are logged to the trace if generation fails
                embedding_obs.end()
                raise e
            finally:
                self.langfuse_tracing.flush()
        else:
            return self.embedding_model.embed_documents(texts)

    async def add_embeddings(self, file_paths: Dict[str, str]) -> None:
        new_documents: List[Document] = []

        # 1. Processing and Chunking
        for path, file_hash in file_paths.items():
            if not self.tracker.add_if_not_exists("processed_pdf_hashes", file_hash):
                continue

            loader: PyPDFLoader = PyPDFLoader(path)
            pages: List[Document] = loader.load()
            chunks: List[Document] = self.text_splitter.split_documents(pages)

            for i, chunk in enumerate(chunks):
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["chunk_id"] = i  
                chunk.metadata["source"] = Path(path).name
                
            new_documents.extend(chunks)

        # 2. Embedding and Vector Store Update
        if new_documents:
            texts: List[str] = [doc.page_content for doc in new_documents]
            metadatas: List[dict] = [doc.metadata for doc in new_documents]
            # embeddings: List[List[float]] = self._generate_embeddings(texts)
            embeddings: List[List[float]] = await run_in_threadpool(self._generate_embeddings, texts)

            payload: dict = {
                "texts": texts,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }

            await self._send_embeddings_with_retry(payload)

    async def _post_to_vector_db_async(self, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url=config.VECTOR_DB_API_URL,
                json=payload,
            )
            response.raise_for_status()

    async def _send_embeddings_with_retry(
        self,
        payload: Dict[str, Any],
        retries: int = 5,
        delay_seconds: float = 2.0,
    ) -> None:
        last_exception: Optional[Exception] = None

        for _ in range(retries):
            try:
                await self._post_to_vector_db_async(payload)
                return
            except Exception as exc:
                last_exception = exc
                await asyncio.sleep(delay_seconds)

        raise RuntimeError(
            f"Failed to send embeddings to vector DB after retries: {last_exception}"
        )