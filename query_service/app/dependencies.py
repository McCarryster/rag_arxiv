from typing import Optional
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

from langfuse import Langfuse
from langfuse import get_client

import config
from vector_search_manager import VectorSearchManager
from redis_cache_manager import RedisCacheManager
from llm_generation_manager import LLMGenerationManager


_vector_search_manager: Optional[VectorSearchManager] = None
_redis_cache_manager: Optional[RedisCacheManager] = None
_llm_generation_manger: Optional[LLMGenerationManager] = None


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

    langfuse = get_client() if config.LANGFUSE_AVAILABLE else None

    _vector_search_manager = VectorSearchManager(
        embedding_model=embedding_model,
        index_name="arxiv_papers_index",
        top_k=config.TOP_K,
        langfuse_tracing=langfuse
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


def get_llm_generation_manger(recreate: bool = False) -> LLMGenerationManager:
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

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    langfuse = get_client() if config.LANGFUSE_AVAILABLE else None

    _llm_generation_manger = LLMGenerationManager(
        client=client,
        langfuse_tracing=langfuse
    )

    return _llm_generation_manger