import os
import shutil
import tempfile
import time
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from opentelemetry.metrics import Counter, Histogram

from monitoring import setup_monitoring, get_meter
from dependencies import get_vector_store_manager, get_pdf_store_manager

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
    vector_store_manager = get_vector_store_manager()
    pdf_store_manager = get_pdf_store_manager()
    
    temp_paths: List[str] = []
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
            temp_paths.append(temp_path)
            processed_files.append(file.filename)

        # Index and get hashes
        path_to_hashes: Dict[str, str] = vector_store_manager.add_pdfs(temp_paths)

        # Store PDFs using the already calculated hashes
        for path in temp_paths:
            file_hash: Optional[str] = path_to_hashes.get(path)
            pdf_store_manager.save_pdf(path, existing_hash=file_hash)
            
            indexed_pdf_counter.add(1, {"status": "success"}) # Increment the Prometheus counter for each successful file

    except Exception as e:
        indexed_pdf_counter.add(len(files), {"status": "failure"}) # Increment failure count if something goes wrong
        raise e

    finally:
        shutil.rmtree(temp_dir) # Cleanup all temp files
        end_time: float = time.perf_counter()
        duration: float = end_time - start_time
        indexing_latency.record(duration, {"endpoint": "/index"}) # Track time
        if hasattr(vector_store_manager, 'monitoring_handler') and vector_store_manager.monitoring_handler:
            vector_store_manager.monitoring_handler.flush() # Flush Langfuse traces specifically

    return IndexResponse(
        message="Process complete",
        files_processed=processed_files
    )