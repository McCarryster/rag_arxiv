from typing import List, Dict, Any, Optional
import time

from pydantic import BaseModel
from fastapi import FastAPI

from langchain_core.documents import Document

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from telemetry import setup_monitoring, get_meter
from opentelemetry.metrics import Counter, Histogram

from dependencies import get_vector_search_manager, get_redis_cache_manager, get_llm_generation_manger
import config


# Very important. Stops all shit tracing (helps to focus only on model related stuff)
langfuse_tracing: Optional[Langfuse] = None
if config.LANGFUSE_AVAILABLE:
    langfuse_tracer_provider = TracerProvider()
    langfuse_tracing = Langfuse(
        blocked_instrumentation_scopes=["fastapi", "starlette"],
        tracer_provider=langfuse_tracer_provider
    )


# --- Models ---
class QueryRequest(BaseModel):
    user_query: str

class IndexResponse(BaseModel):
    message: str
    model_response: str
    sources: List[Dict[str, Any]]


app = FastAPI(title="Arxiv Query Service")


# Initialize Monitoring. This instruments FastAPI and sets up the /metrics endpoint
setup_monitoring(app, service_name="pdf-querying-service")

# Define Metrics
meter = get_meter("pdf_querier")
hb_search_latency: Histogram = meter.create_histogram(
    name="hb_duration_seconds",
    description="Time spent on vector similarity and BM25 search for relevant docs",
    unit="s"
)

redis_semantic_search_latency: Histogram = meter.create_histogram(
    name="redis_semantic_search_duration_seconds",
    description="Time spent on semantic search for a similar query in Redis using KNN",
    unit="s"
)

# --- Endpoint ---
@app.post("/query", response_model=IndexResponse)
async def query_index(request: QueryRequest) -> IndexResponse:
    # 1. Setup Managers
    vector_search_manager = get_vector_search_manager(False, langfuse_tracing)
    cache_manager = get_redis_cache_manager()
    llm_generation_manager = get_llm_generation_manger(False, langfuse_tracing)

    # Caching 1. Exact match caching (skips LLM generation)
    exact_hit = cache_manager.get_exact_cache(request.user_query)
    if exact_hit:
        return IndexResponse(
            message="Success (Exact Cache Hit)",
            model_response=exact_hit["model_response"],
            sources=exact_hit["sources"]
        )


    # 2. Generate Embedding
    query_vector: List[float] = vector_search_manager.generate_embeddings(request.user_query)


    # Caching 2. Similarity caching (skips Hybrid Search)
    # High threshold (0.05) = Skip LLM | (0.15) = Skip Retrieval but re-run LLM
    start_time: float = time.perf_counter()
    semantic_hit = cache_manager.get_semantic_cache(query_vector, threshold=0.15)
    end_time: float = time.perf_counter()
    hb_search_latency.record(end_time-start_time, {"endpoint": "/index"}) # Track time
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
        cached_metadata = data["sources"]
        retrieved_docs = vector_search_manager.get_docs_by_metadata(cached_metadata)


    # 3. If no cache hits for retrieval, perform Hybrid Search
    if not retrieved_docs:
        start_time: float = time.perf_counter()
        retrieved_docs = vector_search_manager.perform_hybrid_search(request.user_query)
        end_time: float = time.perf_counter()
        redis_semantic_search_latency.record(end_time-start_time, {"endpoint": "/index"}) # Track time
    
    if not retrieved_docs:
        return IndexResponse(
            message="No matches",
            model_response="Insufficient relevant details in retrieved papers.",
            sources=[]
        )


    # Execute LLM (Only reached if Exact hit failed AND Semantic high-confidence hit failed)
    answer: str = llm_generation_manager.generate_answer(request.user_query, retrieved_docs)

    # 4. Final Source Collection
    sources: List[Dict[str, Any]] = [
        {
            "id": d.metadata.get("source"),
            "chunk": d.metadata.get("chunk_id"),
            "hash": d.metadata.get("file_hash")
        } 
        for d in retrieved_docs
    ]

    # 5. Store in both Caches
    cache_payload: Dict[str, Any] = {
        "model_response": answer,
        "sources": sources
    }
    
    cache_manager.set_exact_cache(request.user_query, cache_payload) # Store exact text hash
    
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

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}