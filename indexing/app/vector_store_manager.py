import os
import hashlib
from pathlib import Path
from pydantic import SecretStr
from typing import List, Protocol, Optional, runtime_checkable, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dependencies import get_redis_manager
from redis_db import RedisManager
import config


@runtime_checkable
class VectorStorageProvider(Protocol):
    """
    Protocol defining how the vector store should be persisted.
    This allows swapping between Local Storage, AWS S3, or Azure Blob.
    """
    def save(self, vector_store: FAISS, index_name: str) -> None: ...
    def load(self, embeddings: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]: ...
    def exists(self, index_name: str) -> bool: ...

# Storage providers
class LocalStorageProvider:
    def __init__(self, base_path: str) -> None:
        self.base_path: str = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save(self, vector_store: FAISS, index_name: str) -> None:
        save_path: str = os.path.join(self.base_path, index_name)
        vector_store.save_local(save_path)

    def load(self, embeddings: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]:
        load_path: str = os.path.join(self.base_path, index_name)
        if self.exists(index_name):
            return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        return None

    def exists(self, index_name: str) -> bool:
        return os.path.exists(os.path.join(self.base_path, index_name))

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

# Manager for vector db operations
class VectorStoreManager:
    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        storage_provider: Any,
        duplicate_tracker: RedisManager,
        index_name: str
    ) -> None:
        self.embeddings: OpenAIEmbeddings = embedding_model
        self.text_splitter: RecursiveCharacterTextSplitter = text_splitter
        self.storage = storage_provider
        self.tracker: RedisManager = duplicate_tracker
        self.index_name: str = index_name
        self.vector_store: Optional[FAISS] = self.storage.load(self.embeddings, self.index_name)

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

    def add_pdfs(self, file_paths: List[str]) -> None:
        new_documents: List[Document] = []

        for path in file_paths:
            file_hash: str = self._get_file_hash(path)
            
            if self.tracker.is_duplicate("processed_pdf_hashes", file_hash):
                continue

            loader = PyPDFLoader(path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            
            for chunk in chunks:
                chunk.metadata["file_hash"] = file_hash
                
            new_documents.extend(chunks)
            self.tracker.add_hash("processed_pdf_hashes", file_hash)

        if new_documents:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(new_documents, self.embeddings)
            else:
                self.vector_store.add_documents(new_documents)
            
            self.storage.save(self.vector_store, self.index_name)


def get_vector_store_manager() -> VectorStoreManager:
    embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_EMBEDDING_MODEL
    )
    
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )

    # Injected dependencies
    if not config.PROD:
        storage_provider = LocalStorageProvider(base_path="/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/vector_db")
    else:
        ...

    # Redis instance
    redis_manager = get_redis_manager()

    return VectorStoreManager(
        embedding_model=embedding_model,
        text_splitter=text_splitter,
        storage_provider=storage_provider,
        duplicate_tracker=redis_manager,
        index_name="arxiv_papers_index"
    )