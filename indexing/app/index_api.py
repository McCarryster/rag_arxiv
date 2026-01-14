import os
import shutil
import tempfile
import time
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from opentelemetry.metrics import Counter, Histogram

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider

from telemetry import setup_monitoring, get_meter
from dependencies import get_vector_store_manager, get_pdf_store_manager, get_redis_manager
import config


# Very important. Stops all shit tracing (helps to focus only on model related stuff)
langfuse_tracing: Optional[Langfuse] = None
if config.LANGFUSE_AVAILABLE:
    langfuse_tracer_provider = TracerProvider()
    langfuse_tracing = Langfuse(
        blocked_instrumentation_scopes=["fastapi", "starlette"],
        tracer_provider=langfuse_tracer_provider
    )

class IndexResponse(BaseModel):
    message: str
    files_processed: List[str]

app: FastAPI = FastAPI(title="PDF Indexing Service")

# Initialize Monitoring. This instruments FastAPI and sets up the /metrics endpoint
setup_monitoring(app, service_name="pdf-indexing-service")

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


@app.post("/index", response_model=IndexResponse)
async def index_data(files: List[UploadFile] = File(...)) -> IndexResponse:
    start_time: float = time.perf_counter()

    # Setup managers
    vector_store_manager = get_vector_store_manager(False, langfuse_tracing)
    pdf_store_manager = get_pdf_store_manager()
    redis_manager = get_redis_manager()
    
    # temp_paths: List[str] = []
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
        vector_store_manager.add_pdfs(temp_paths_hashes)

        # Store PDFs using the already calculated hashes
        for path, file_hash in temp_paths_hashes.items():
            pdf_store_manager.save_pdf(path, existing_hash=file_hash)
            processed_files.append(path)
            indexed_pdf_counter.add(1, {"status": "success"}) # Increment the Prometheus counter for each successful file

    except Exception as e:
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