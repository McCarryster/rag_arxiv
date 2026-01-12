from typing import Optional
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings

import config
from vector_search_manager import VectorSearchManager, LocalBM25StorageProvider, LocalFaissStorageProvider
from redis_cache_manager import RedisCacheManager

_vector_search_manager: Optional[VectorSearchManager] = None
_redis_cache_manager: Optional[RedisCacheManager] = None


def get_vector_search_manager(recreate: bool = False) -> VectorSearchManager:
    """
    Get or create the singleton VectorSearchManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        VectorSearchManager instance
    """
    global _vector_search_manager

    if _vector_search_manager is not None and not recreate:
        return _vector_search_manager

    # 1. Setup Model
    embedding_model = OpenAIEmbeddings(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_EMBEDDING_MODEL
    )

    # 2. Setup Storages (Environment Aware)
    if not config.PROD:
        faiss_p = LocalFaissStorageProvider(base_path=config.LOCAL_FAISS_PATH)
        bm25_p = LocalBM25StorageProvider(base_path=config.LOCAL_BM25_PATH)
    else:
        raise NotImplementedError("S3 Storage Providers not yet implemented for PROD.")

    _vector_search_manager = VectorSearchManager(
        embedding_model=embedding_model,
        faiss_storage_provider=faiss_p,
        bm25_storage_provider=bm25_p,
        index_name="arxiv_papers_index",
        top_k=config.TOP_K
    )

    return _vector_search_manager

def get_redis_cache_manager(recreate: bool = False) -> RedisCacheManager:
    """
    Get or create the singleton RedisCacheManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        RedisCacheManager instance
    """
    global _redis_cache_manager
    # Only create if it doesn't exist OR if the path has changed (important for tests!)
    current_redis_config = config.RD
    
    if _redis_cache_manager is None or _redis_cache_manager.redis_config != current_redis_config or recreate:
        _redis_cache_manager = RedisCacheManager(current_redis_config)
    
    return _redis_cache_manager