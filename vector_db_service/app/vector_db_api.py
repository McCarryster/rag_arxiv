from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from vector_db_manager import WeaviateDBManager
from dependencies import (
    init_vector_db_manager,
    close_vector_db_manager,
    get_vector_db_manager,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    """
    init_vector_db_manager()
    try:
        yield
    finally:
        close_vector_db_manager()

app: FastAPI = FastAPI(title="Weaviate VectorDB API", lifespan=lifespan)


# -------------------------
# Models
# -------------------------
class AddEmbeddingsRequest(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]

class HybridSearchRequest(BaseModel):
    query_text: str = Field(..., description="Raw query text for BM25")
    query_embedding: List[float] = Field(..., description="Query embedding")
    limit: int = Field(default=5, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

class HybridSearchResult(BaseModel):
    id: str
    text: str
    file_hash: str
    chunk_id: int

class HybridSearchResponse(BaseModel):
    results: List[HybridSearchResult]


# -------------------------
# Routes
# -------------------------
@app.post("/embeddings/add")
def add_embeddings(
    request: AddEmbeddingsRequest,
    vector_db_manager: WeaviateDBManager = Depends(get_vector_db_manager),
) -> Dict[str, str]:
    try:
        vector_db_manager.add_embeddings(
            texts=request.texts,
            embeddings=request.embeddings,
            metadatas=request.metadatas,
        )
    except ValueError as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to add embeddings") from exc
    
    return {"status": "success"}

@app.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    request: HybridSearchRequest,
    vector_db_manager: WeaviateDBManager = Depends(get_vector_db_manager),
) -> HybridSearchResponse:

    try:
        results = vector_db_manager.hybrid_search(
            query_text=request.query_text,
            query_embedding=request.query_embedding,
            limit=request.limit,
            alpha=request.alpha,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {exc}")

    return HybridSearchResponse(
        results=[
            HybridSearchResult(
                id=str(item["id"]),
                text=item["text"],
                file_hash=item["file_hash"],
                chunk_id=item["chunk_id"],
            )
            for item in results
        ]
    )

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}