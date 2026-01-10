from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import tempfile
import shutil
import os
from dependencies import get_vector_store_manager, get_pdf_store_manager


# Test
# folder: Path = Path("/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/local_arxiv_pdfs")
# pdf_paths: List[str] = [str(path) for path in folder.iterdir() if path.is_file()]


# vector_store_manager = get_vector_store_manager()
# vector_store_manager.add_pdfs(pdf_paths)

# Add endpoint and stuff..

# 1. Take folder with pdfs or a single file
# 2. Save that new pdf in folder
# 3. Make vectorization and storing to faiss
# 4. Make logging and stuff



class IndexResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks_added: Optional[int] = None
app = FastAPI(title="PDF Indexing Service")

@app.post("/index", response_model=IndexResponse)
async def index_data(files: List[UploadFile] = File(...)) -> IndexResponse:
    # 1. Setup managers
    vector_store_manager = get_vector_store_manager()
    pdf_store_manager = get_pdf_store_manager()
    
    temp_paths: List[str] = []
    processed_files: List[str] = []

    temp_dir = tempfile.mkdtemp()
    try:
        # 2. Save uploads to temporary disk locations
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue
            
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(temp_path)
            processed_files.append(file.filename)

        # 3. Index and get hashes
        path_to_hashes: dict[str, str] = vector_store_manager.add_pdfs(temp_paths)

        # 4. Store PDFs using the already calculated hashes
        for path in temp_paths:
            file_hash = path_to_hashes.get(path)
            pdf_store_manager.save_pdf(path, existing_hash=file_hash)

    finally:
        # Cleanup all temp files
        shutil.rmtree(temp_dir)

    return IndexResponse(
        message="Process complete",
        files_processed=processed_files
    )