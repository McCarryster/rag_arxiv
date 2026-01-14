from typing import Optional
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langfuse import Langfuse

import config
from redis_manager import RedisManager
from pdf_store_manager import PDFStoreManager, LocalPDFStorage
from vector_store_manager import VectorStoreManager, LocalBM25StorageProvider, LocalFaissStorageProvider

_redis_manager: Optional[RedisManager] = None
_vector_store_manager: Optional[VectorStoreManager] = None
_pdf_store_manager: Optional[PDFStoreManager] = None


def get_redis_manager(recreate: bool = False) -> RedisManager:
    """
    Get or create the singleton RedisManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        RedisManager instance
    """
    global _redis_manager
    # Only create if it doesn't exist OR if the path has changed (important for tests!)
    current_redis_config = config.RD
    
    if _redis_manager is None or _redis_manager.redis_config != current_redis_config or recreate:
        _redis_manager = RedisManager(current_redis_config)
    
    return _redis_manager


def get_vector_store_manager(recreate: bool = False, langfuse_tracing: Optional[Langfuse] = None) -> VectorStoreManager:
    """
    Get or create the singleton VectorStoreManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        VectorStoreManager instance
    """
    global _vector_store_manager

    if _vector_store_manager is not None and not recreate:
        return _vector_store_manager

    # 1. Setup Models
    embedding_model = OpenAIEmbeddings(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_EMBEDDING_MODEL
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )

    # 2. Setup Storage (Environment Aware)
    if not config.PROD:
        faiss_p = LocalFaissStorageProvider(base_path=config.LOCAL_FAISS_PATH)
        bm25_p = LocalBM25StorageProvider(base_path=config.LOCAL_BM25_PATH)
    else:
        raise NotImplementedError("S3 Storage Providers not yet implemented for PROD.")

    # 3. Setup Redis
    redis_manager = get_redis_manager(recreate=recreate)

    _vector_store_manager = VectorStoreManager(
        embedding_model=embedding_model,
        text_splitter=text_splitter,
        faiss_storage_provider=faiss_p,
        bm25_storage_provider=bm25_p,
        duplicate_tracker=redis_manager,
        index_name="arxiv_papers_index",
        langfuse_tracing=langfuse_tracing
    )

    return _vector_store_manager


def get_pdf_store_manager(recreate: bool = False) -> PDFStoreManager:
    """
    Get or create the singleton PDFStoreManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        PDFStoreManager instance
    """

    global _pdf_store_manager
    if _pdf_store_manager is not None and not recreate:
        return _pdf_store_manager

    if not config.PROD:
        pdf_storage_provider = LocalPDFStorage(base_dir=config.LOCAL_PDF_STORAGE_PATH)
    else:
        raise NotImplementedError("S3 Storage Providers not yet implemented for PROD.")

    return PDFStoreManager(storage_provider=pdf_storage_provider)