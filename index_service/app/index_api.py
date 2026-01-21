import os
import shutil
import tempfile
import time
from typing import List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from opentelemetry.metrics import Counter, Histogram

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from telemetry import setup_monitoring, get_meter

from vector_store_manager import VectorStoreManager
from pdf_store_manager import PDFStoreManager
from redis_manager import RedisManager
from dependencies import get_vector_store_manager, get_pdf_store_manager, get_redis_manager

import config


# Very important. Stops tracing all shit (helps to focus only on model related stuff)
langfuse_tracing = None
if config.LANGFUSE_AVAILABLE:
    langfuse_tracer_provider = TracerProvider()
    langfuse_tracing = Langfuse(
        blocked_instrumentation_scopes=["fastapi", "starlette"],
        tracer_provider=langfuse_tracer_provider
    )

# ------------------------------
# Models
# ------------------------------
class IndexResponse(BaseModel):
    message: str
    files_processed: List[str]


# ------------------------------
# FastAPI lifespan for pre-initializing singletons
# ------------------------------
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Initialize singleton managers at startup
    get_redis_manager()
    get_vector_store_manager()
    get_pdf_store_manager()
    yield
    # Optional cleanup can be added here if needed


# ------------------------------
# FastAPI app
# ------------------------------
app: FastAPI = FastAPI(title="PDF Indexing Service", lifespan=app_lifespan)


# ------------------------------
# Monitoring setup
# ------------------------------
setup_monitoring(app, service_name="pdf-indexing-service") # Initialize Monitoring. Instruments FastAPI and sets up the /metrics endpoint
# Define Metrics
meter = get_meter("pdf_indexer")
# Counter to track how many PDFs are successfully indexed
indexed_pdf_counter: Counter = meter.create_counter(
    name="pdfs_indexed_total",
    description="Total number of PDF files successfully processed and indexed",
    unit="1"
)
# Histogram for latency
indexing_latency: Histogram = meter.create_histogram(
    name="pdf_indexing_duration_seconds",
    description="Time spent indexing PDF files",
    unit="s"
)


# ------------------------------
# FastAPI app
# ------------------------------
@app.post("/index", response_model=IndexResponse)
async def index_data(
    files: List[UploadFile] = File(...),
    vector_store_manager: VectorStoreManager = Depends(get_vector_store_manager),
    pdf_store_manager: PDFStoreManager = Depends(get_pdf_store_manager),
    redis_manager: RedisManager = Depends(get_redis_manager),
) -> IndexResponse:
    start_time: float = time.perf_counter()

    temp_paths_hashes: Dict[str, str] = {} # {temp_path: file_hash}
    processed_files: List[str] = []
    temp_dir: str = tempfile.mkdtemp()

    try:
        # Save uploads to temporary disk locations
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue
            temp_path: str = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            current_hash: str = vector_store_manager.get_file_hash(temp_path)
            if not redis_manager.is_duplicate("processed_pdf_hashes", current_hash):
                temp_paths_hashes[temp_path] = current_hash

        # Index and get hashes
        vector_store_manager.add_embeddings(temp_paths_hashes)

        # Store PDFs using the already calculated hashes
        for path, file_hash in temp_paths_hashes.items():
            pdf_store_manager.save_pdf(path, existing_hash=file_hash)
            processed_files.append(path)
            indexed_pdf_counter.add(1, {"status": "success"}) # Increment the Prometheus counter for each successful file

    except Exception as e:
        import traceback
        traceback.print_exc()
        indexed_pdf_counter.add(len(files), {"status": "failure"}) # Increment failure count if something goes wrong
        raise e

    finally:
        shutil.rmtree(temp_dir) # Cleanup all temp files
        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        indexing_latency.record(duration, {"endpoint": "/index"}) # Track time

    return IndexResponse(
        message="Process complete",
        files_processed=processed_files
    )


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}