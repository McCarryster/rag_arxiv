import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Protocol, Optional, runtime_checkable, Any
import tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from langfuse import Langfuse

from redis_manager import RedisManager
import config

# Protocols for providers
@runtime_checkable
class VectorStorageProvider(Protocol):
    def save(self, vector_store: FAISS, index_name: str) -> None: ...
    def load(self, embedding_model: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]: ...
    def exists(self, index_name: str) -> bool: ...

@runtime_checkable
class BM25StorageProvider(Protocol):
    def save(self, retriever: BM25Retriever, index_name: str) -> None: ...
    def load(self, index_name: str) -> Optional[BM25Retriever]: ...
    def exists(self, index_name: str) -> bool: ... # Added for full symmetry


# Storage providers
class LocalFaissStorageProvider:
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, vector_store: FAISS, index_name: str) -> None:
        save_path: Path = self.base_path / index_name
        vector_store.save_local(str(save_path))

    def load(self, embedding_model: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]:
        load_path: Path = self.base_path / index_name
        if self.exists(index_name):
            return FAISS.load_local(
                str(load_path), 
                embedding_model,
                allow_dangerous_deserialization=True
            )
        return None

    def exists(self, index_name: str) -> bool:
        return (self.base_path / index_name).exists()

class LocalBM25StorageProvider:
    """Handles saving/loading/checking the BM25 sparse index."""
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, retriever: BM25Retriever, index_name: str) -> None:
        save_path: Path = self.base_path / f"{index_name}_bm25.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(retriever, f)

    def load(self, index_name: str) -> Optional[BM25Retriever]:
        if self.exists(index_name):
            load_path: Path = self.base_path / f"{index_name}_bm25.pkl"
            with open(load_path, "rb") as f:
                return pickle.load(f)
        return None

    def exists(self, index_name: str) -> bool:
        return (self.base_path / f"{index_name}_bm25.pkl").exists()


# Manager for vector db operations
class VectorStoreManager:
    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        faiss_storage_provider: Any,
        bm25_storage_provider: Any,
        duplicate_tracker: RedisManager,
        index_name: str,
        langfuse_tracing: Optional[Langfuse] = None
    ) -> None:
        self.embedding_model: OpenAIEmbeddings = embedding_model
        self.text_splitter: RecursiveCharacterTextSplitter = text_splitter
        self.faiss_storage = faiss_storage_provider
        self.bm25_storage = bm25_storage_provider
        self.tracker: RedisManager = duplicate_tracker
        self.index_name: str = index_name
        self.langfuse_tracing = langfuse_tracing

        self.vector_store: Optional[FAISS] = self.faiss_storage.load(self.embedding_model, self.index_name)
        self.bm25_retriever: Optional[BM25Retriever] = self.bm25_storage.load(self.index_name)

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
                self.langfuse_tracing.flush()
                return embeddings
            except Exception as e:
                embedding_obs.update(level="ERROR", status_message=str(e))  # Ensure errors are logged to the trace if generation fails
                embedding_obs.end()
                raise e
        else:
            return self.embedding_model.embed_documents(texts)

    def add_pdfs(self, file_paths: Dict[str, str]) -> None:
        new_documents: List[Document] = []

        # 1. Processing and Chunking
        for path, file_hash in file_paths.items():
            if self.tracker.is_duplicate("processed_pdf_hashes", file_hash):
                continue

            loader: PyPDFLoader = PyPDFLoader(path)
            pages: List[Document] = loader.load()
            chunks: List[Document] = self.text_splitter.split_documents(pages)

            for i, chunk in enumerate(chunks):
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["chunk_id"] = i  
                chunk.metadata["source"] = Path(path).name
                
            new_documents.extend(chunks)
            self.tracker.add_hash("processed_pdf_hashes", file_hash)

        # 2. Embedding and Vector Store Update
        if new_documents:
            texts: List[str] = [doc.page_content for doc in new_documents]
            metadatas: List[dict] = [doc.metadata for doc in new_documents]

            embeddings: List[List[float]] = self._generate_embeddings(texts)
            text_embeddings = list(zip(texts, embeddings))

            # Update FAISS
            if self.vector_store is None:
                self.vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=self.embedding_model, metadatas=metadatas)
            else:
                self.vector_store.add_embeddings(text_embeddings=text_embeddings, metadatas=metadatas)
            self.faiss_storage.save(self.vector_store, self.index_name)

            # 2. Update BM25 (Sparse)
            all_docs: List[Document] = []
            if self.vector_store and isinstance(self.vector_store.docstore, InMemoryDocstore):
                # Pull all current docs from FAISS to ensure BM25 is synced
                doc_dict: Dict[str, Document] = getattr(self.vector_store.docstore, "_dict", {})
                all_docs = list(doc_dict.values())
            if all_docs:
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_storage.save(self.bm25_retriever, self.index_name)
