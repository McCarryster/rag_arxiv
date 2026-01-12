from typing import List, Dict, Any
from pydantic import BaseModel, SecretStr
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dependencies import get_vector_search_manager, get_redis_cache_manager
from utils import format_context_for_prompt
import config
import prompt


# --- Models ---
class QueryRequest(BaseModel):
    user_query: str

class IndexResponse(BaseModel):
    message: str
    model_response: str
    sources: List[Dict[str, Any]]


app = FastAPI(title="Arxiv Query Service")


# --- Endpoint ---
@app.post("/query", response_model=IndexResponse)
async def query_index(request: QueryRequest) -> IndexResponse:
    # 1. Setup Managers
    vector_search_manager = get_vector_search_manager()
    cache_manager = get_redis_cache_manager()


    # Caching 1. Exact match caching (skips LLM generation)
    exact_hit = cache_manager.get_exact_cache(request.user_query)
    if exact_hit:
        return IndexResponse(
            message="Success (Exact Cache Hit)",
            model_response=exact_hit["model_response"],
            sources=exact_hit["sources"]
        )


    # 2. Generate Embedding
    query_vector: List[float] = vector_search_manager.embedding_model.embed_query(request.user_query)


    # Caching 2. Similarity caching (skips Hybrid Search)
    # High threshold (0.05) = Skip LLM. 
    # Mid threshold (0.15) = Skip Retrieval but re-run LLM.
    semantic_hit = cache_manager.get_semantic_cache(query_vector, threshold=0.15)
    
    retrieved_docs: List[Document] = []
    if semantic_hit:
        score: float = semantic_hit["score"]
        data: Dict[str, Any] = semantic_hit["data"]
        
        # Scenario 1: Extremely similar query (High Confidence)
        if score <= 0.05:
            return IndexResponse(
                message=f"Success (Semantic Cache Hit - Score: {score:.4f})",
                model_response=data["model_response"],
                sources=data["sources"]
            )
        
        # Scenario 2: Same topic, different phrasing (Medium Confidence). Reuse the source metadata to skip Hybrid Search
        print(f"Semantic Retrieval Hit (Score: {score:.4f}). Skipping Hybrid Search.")
        cached_metadata = data["sources"]
        retrieved_docs = vector_search_manager.get_docs_by_metadata(cached_metadata)


    # 3. If no cache hits for retrieval, perform Hybrid Search
    if not retrieved_docs:
        retrieved_docs = vector_search_manager.perform_hybrid_search(request.user_query)

    if not retrieved_docs:
        return IndexResponse(
            message="No matches",
            model_response="Insufficient relevant details in retrieved papers.",
            sources=[]
        )

    # 4. Model Setup & Execution
    text_generation_model = ChatOpenAI(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_GENERATION_MODEL,
        temperature=0
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt.SYSTEM_PROMPT),
        ("human", "{query}")
    ])

    chain = (
        {
            "context": lambda x: format_context_for_prompt(retrieved_docs),
            "query": RunnablePassthrough()
        }
        | prompt_template
        | text_generation_model
        | StrOutputParser()
    )

    # Execute LLM (Only reached if Exact hit failed AND Semantic high-confidence hit failed)
    answer: str = await chain.ainvoke(request.user_query)

    # 5. Final Source Collection
    sources: List[Dict[str, Any]] = [
        {
            "id": d.metadata.get("source"),
            "chunk": d.metadata.get("chunk_id"),
            "hash": d.metadata.get("file_hash")
        } 
        for d in retrieved_docs
    ]

    # 6. Store in both Caches
    cache_payload: Dict[str, Any] = {
        "model_response": answer,
        "sources": sources
    }
    
    # Store exact text hash
    cache_manager.set_exact_cache(request.user_query, cache_payload)
    
    # Store semantic vector
    cache_manager.set_semantic_cache(
        query_text=request.user_query,
        query_vector=query_vector,
        data=cache_payload
    )

    return IndexResponse(
        message="Success",
        model_response=answer,
        sources=sources
    )