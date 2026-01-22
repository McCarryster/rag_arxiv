from typing import List, Dict, Any, Optional
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel
from fastapi import FastAPI, Depends

from langchain_core.documents import Document

from langfuse import Langfuse
from langfuse import get_client
from opentelemetry.sdk.trace import TracerProvider
from telemetry import setup_monitoring, get_meter
from opentelemetry.metrics import Counter, Histogram


from dependencies import get_vector_search_manager, get_redis_cache_manager, get_llm_generation_manger
from vector_search_manager import VectorSearchManager
from llm_generation_manager import LLMGenerationManager
from redis_cache_manager import RedisCacheManager
import config


# Very important. Stops all shit tracing (helps to focus only on model related stuff)
langfuse_tracing: Optional[Langfuse] = None
if config.LANGFUSE_AVAILABLE:
    langfuse_tracer_provider = TracerProvider()
    langfuse_tracing = Langfuse(
        blocked_instrumentation_scopes=["fastapi", "starlette"],
        tracer_provider=langfuse_tracer_provider
    )
langfuse = get_client()

# ------------------------------
# Models
# ------------------------------
class QueryRequest(BaseModel):
    user_query: str

class IndexResponse(BaseModel):
    message: str
    model_response: str
    sources: List[Dict[str, Any]]


# ------------------------------
# FastAPI lifespan for pre-initializing singletons
# ------------------------------
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Initialize singleton managers at startup
    get_vector_search_manager()
    get_redis_cache_manager()
    get_llm_generation_manger()
    yield
    # Optional cleanup can be added here if needed

app: FastAPI = FastAPI(title="Arxiv Query Service", lifespan=app_lifespan)


# ------------------------------
# Monitoring setup
# ------------------------------
setup_monitoring(app, service_name="pdf-querying-service") # Initialize Monitoring. This instruments FastAPI and sets up the /metrics endpoint
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


# ------------------------------
# FastAPI app
# ------------------------------
@app.post("/query", response_model=IndexResponse)
async def query_index(
    request: QueryRequest,
    vector_search_manager: VectorSearchManager = Depends(get_vector_search_manager),
    cache_manager: RedisCacheManager = Depends(get_redis_cache_manager),
    llm_generation_manager: LLMGenerationManager = Depends(get_llm_generation_manger),
) -> IndexResponse:
    with langfuse.start_as_current_observation(as_type="span", name="rag-query-pipeline") as parent_span:
        # ------------------------------------------------------------------
        # Corpus version (changes whenever documents are ingested/updated)
        # ------------------------------------------------------------------
        corpus_version: int = cache_manager.get_corpus_version()


        # ------------------------------------------------------------------
        # 1. Exact cache (ONLY valid if corpus version matches)
        # ------------------------------------------------------------------
        exact_hit: Optional[Dict[str, Any]] = cache_manager.get_exact_cache(request.user_query, corpus_version=corpus_version)
        if exact_hit:
            print("[DEBUG]:exact hit", request.user_query, flush=True)
            return IndexResponse(
                message="Success (Exact Cache Hit)",
                model_response=exact_hit["model_response"],
                sources=exact_hit["sources"]
            )


        # ------------------------------------------------------------------
        # 2. Generate query embedding
        # ------------------------------------------------------------------
        query_embedding: List[float] = await vector_search_manager.generate_embeddings(request.user_query)


        # ------------------------------------------------------------------
        # 3. Semantic cache (validated by corpus version + non-empty sources)
        # ------------------------------------------------------------------
        start_time: float = time.perf_counter()
        semantic_hit: Optional[Dict[str, Any]] = cache_manager.get_semantic_cache(
            query_vector=query_embedding,
            threshold=0.7,
            corpus_version=corpus_version,
        )
        print("[DEBUG]:semantic hit", request.user_query, flush=True)
        end_time: float = time.perf_counter()
        hb_search_latency.record(end_time-start_time, {"endpoint": "/index"}) # Track time

        retrieved_docs: List[Document] = []
        if semantic_hit:
            score: float = semantic_hit["score"]
            data: Dict[str, Any] = semantic_hit["data"]
            print("[DEBUG]:score", score, flush=True)
            
            # Never reuse semantic cache created with NO retrieval
            if not data.get("sources"):
                semantic_hit = None
            else:
                # Scenario 1: Extremely high confidence -> reuse full answer
                # if score <= 0.05:
                if score <= 0.2:
                    return IndexResponse(
                        message=f"Success (Semantic Cache Hit - Score: {score:.4f})",
                        model_response=data["model_response"],
                        sources=data["sources"],
                    )
            
            # Scenario 2: Same topic, different phrasing (Medium Confidence). Reuse the source metadata to skip Hybrid Search
            cached_metadata: List[Dict[str, Any]] = data["sources"]
            # print("DEBUG data['sources']:", cached_metadata, flush=True)
            retrieved_docs = await vector_search_manager.get_docs_by_metadata(cached_metadata)
            print("[DEBUG] retrieved_docs:", retrieved_docs, flush=True)


        # ------------------------------------------------------------------
        # 4. Hybrid search fallback
        # ------------------------------------------------------------------
        if not retrieved_docs:
            start_time: float = time.perf_counter()
            retrieved_docs = await vector_search_manager.perform_hybrid_search(request.user_query, query_embedding)
            end_time: float = time.perf_counter()
            redis_semantic_search_latency.record(end_time-start_time, {"endpoint": "/query"}) # Track time
        else:
            print("[DEBUG]:", "HYBRID SEARCH SKIPPED", flush=True)
        

        # ------------------------------------------------------------------
        # 5. No documents -> return response BUT DO NOT SEMANTIC CACHE
        # ------------------------------------------------------------------
        if not retrieved_docs:
            return IndexResponse(
                message="No matches",
                model_response="Insufficient relevant details in retrieved papers.",
                sources=[]
            )


        # ------------------------------------------------------------------
        # 6. Execute LLM
        # ------------------------------------------------------------------
        answer: str = await llm_generation_manager.generate_answer(request.user_query, retrieved_docs)


        # ------------------------------------------------------------------
        # 7. Collect sources
        # ------------------------------------------------------------------
        sources: List[Dict[str, Any]] = [
            {
                "id": d.metadata.get("id"),
                "chunk_id": d.metadata.get("chunk_id"),
                "file_hash": d.metadata.get("file_hash")
            } 
            for d in retrieved_docs
        ]


        # ------------------------------------------------------------------
        # 8. Cache ONLY authoritative answers (with docs + corpus version)
        # ------------------------------------------------------------------
        cache_payload: Dict[str, Any] = {
            "model_response": answer,
            "sources": sources,
            "corpus_version": corpus_version,
        }

        cache_manager.set_exact_cache(
            request.user_query,
            cache_payload,
            corpus_version=corpus_version,
        )
        
        cache_manager.set_semantic_cache(
            query_text=request.user_query,
            query_vector=query_embedding,
            data=cache_payload,
            corpus_version=corpus_version,
        )
        parent_span.update(output={"answer": answer})

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