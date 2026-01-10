import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Protocol, Optional, runtime_checkable, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from redis_db import RedisManager


@runtime_checkable
class VectorStorageProvider(Protocol):
    """
    Protocol defining how the vector store should be persisted.
    This allows swapping between Local Storage, AWS S3, or Azure Blob.
    """
    def save(self, vector_store: FAISS, index_name: str) -> None: ...
    def load(self, embeddings: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]: ...
    def exists(self, index_name: str) -> bool: ...

@runtime_checkable
class BM25StorageProvider(Protocol):
    """Protocol for BM25 storage."""
    def save(self, retriever: BM25Retriever, index_name: str) -> None: ...
    def load(self, index_name: str) -> Optional[BM25Retriever]: ...

# Faiss storage providers
class LocalFaissStorageProvider:
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, vector_store: FAISS, index_name: str) -> None:
        save_path: Path = self.base_path / index_name
        vector_store.save_local(str(save_path))

    def load(self, embeddings: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]:
        load_path: Path = self.base_path / index_name
        if self.exists(index_name):
            # FAISS expects a string path
            return FAISS.load_local(
                str(load_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

    def exists(self, index_name: str) -> bool:
        return (self.base_path / index_name).exists()

# class S3StorageProvider:
#     def __init__(self, bucket_name: str, s3_prefix: str = "vector-indices/") -> None:
#         self.s3 = boto3.client('s3')
#         self.bucket: str = bucket_name
#         self.prefix: str = s3_prefix
#         self.local_tmp: str = "/tmp/faiss_cache"

#     def save(self, vector_store: FAISS, index_name: str) -> None:
#         # 1. Save locally first
#         local_path = Path(self.local_tmp) / index_name
#         vector_store.save_local(str(local_path))
        
#         # 2. Upload files to S3 (FAISS creates an index.faiss and index.pkl)
#         for file in local_path.iterdir():
#             s3_key = f"{self.prefix}{index_name}/{file.name}"
#             self.s3.upload_file(str(file), self.bucket, s3_key)

#     def load(self, embeddings: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]:
#         local_path = Path(self.local_tmp) / index_name
#         local_path.mkdir(parents=True, exist_ok=True)
        
#         try:
#             # Download files from S3 to local /tmp
#             response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}{index_name}/")
#             if 'Contents' not in response:
#                 return None
                
#             for obj in response['Contents']:
#                 file_name = obj['Key'].split('/')[-1]
#                 self.s3.download_file(self.bucket, obj['Key'], str(local_path / file_name))
            
#             return FAISS.load_local(str(local_path), embeddings, allow_dangerous_deserialization=True)
#         except Exception:
#             return None

#     def exists(self, index_name: str) -> bool:
#         response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}{index_name}/", MaxKeys=1)
#         return 'Contents' in response


# BM25 storage providers
class LocalBM25StorageProvider:
    """Handles saving/loading the BM25 sparse index."""
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, retriever: BM25Retriever, index_name: str) -> None:
        save_path: Path = self.base_path / f"{index_name}_bm25.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(retriever, f)

    def load(self, index_name: str) -> Optional[BM25Retriever]:
        load_path: Path = self.base_path / f"{index_name}_bm25.pkl"
        if load_path.exists():
            with open(load_path, "rb") as f:
                return pickle.load(f)
        return None


# Manager for vector db operations
class VectorStoreManager:
    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        faiss_storage_provider: Any,
        bm25_storage_provider: Any,
        duplicate_tracker: RedisManager,
        index_name: str
    ) -> None:
        self.embedding_model: OpenAIEmbeddings = embedding_model
        self.text_splitter: RecursiveCharacterTextSplitter = text_splitter
        self.faiss_storage = faiss_storage_provider
        self.bm25_storage = bm25_storage_provider
        self.tracker: RedisManager = duplicate_tracker
        self.index_name: str = index_name

        self.vector_store: Optional[FAISS] = self.faiss_storage.load(self.embedding_model, self.index_name)
        self.bm25_retriever: Optional[BM25Retriever] = self.bm25_storage.load(self.index_name)

    def _get_file_hash(self, file_path: str) -> str:
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

    def add_pdfs(self, file_paths: List[str]) -> Dict[str, str]:
        new_documents: List[Document] = []
        path_to_hash_map: Dict[str, str] = {}

        for path in file_paths:
            file_hash: str = self._get_file_hash(path)
            path_to_hash_map[path] = file_hash

            if self.tracker.is_duplicate("processed_pdf_hashes", file_hash):
                continue

            loader: PyPDFLoader = PyPDFLoader(path)
            pages: List[Document] = loader.load()
            chunks: List[Document] = self.text_splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["file_hash"] = file_hash
                
            new_documents.extend(chunks)
            self.tracker.add_hash("processed_pdf_hashes", file_hash)

        if new_documents:
            # 1. Update FAISS (Dense)
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(new_documents, self.embedding_model)
            else:
                self.vector_store.add_documents(new_documents)
            self.faiss_storage.save(self.vector_store, self.index_name)

            # 2. Update BM25 (Sparse)
            all_docs: List[Document] = []
            if self.vector_store and isinstance(self.vector_store.docstore, InMemoryDocstore):
                # Pull all current docs from FAISS to ensure BM25 is synced
                doc_dict = getattr(self.vector_store.docstore, "_dict", {})
                all_docs = list(doc_dict.values())
            if all_docs:
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_storage.save(self.bm25_retriever, self.index_name)

        return path_to_hash_map