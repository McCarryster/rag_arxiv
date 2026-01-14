from typing import Optional
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langfuse import Langfuse

import config
import prompt
from utils import format_context_for_prompt
from vector_search_manager import VectorSearchManager, LocalBM25StorageProvider, LocalFaissStorageProvider
from redis_cache_manager import RedisCacheManager
from llm_generation_manager import LLMGenerationManager

_vector_search_manager: Optional[VectorSearchManager] = None
_redis_cache_manager: Optional[RedisCacheManager] = None
_llm_generation_manger: Optional[LLMGenerationManager] = None

def get_vector_search_manager(recreate: bool = False, langfuse_tracing: Optional[Langfuse] = None) -> VectorSearchManager:
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
        top_k=config.TOP_K,
        langfuse_tracing=langfuse_tracing
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

def get_llm_generation_manger(recreate: bool = False, langfuse_tracing: Optional[Langfuse] = None) -> LLMGenerationManager:
    """
    Get or create the singleton LLMGenerationManager instance.
    
    Args:
        recreate: Force creation of a new instance even if one exists
        
    Returns:
        LLMGenerationManager instance
    """
    global _llm_generation_manger

    if _llm_generation_manger is not None and not recreate:
        return _llm_generation_manger
    
    text_generation_model = ChatOpenAI(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_GENERATION_MODEL,
        temperature=config.TEMPERATURE 
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt.SYSTEM_PROMPT),
        ("human", "{query}")
    ])

    _llm_generation_manger = LLMGenerationManager(
        text_generation_model=text_generation_model,
        prompt_template=prompt_template,
        langfuse_tracing=langfuse_tracing
    )

    return _llm_generation_manger