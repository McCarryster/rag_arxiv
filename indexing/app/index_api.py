from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path

from vector_store_manager import get_vector_store_manager
from dependencies import get_redis_manager


# Test
folder: Path = Path("/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/local_arxiv_pdfs")
pdf_paths: List[str] = [str(path) for path in folder.iterdir() if path.is_file()]


vector_store_manager = get_vector_store_manager()
vector_store_manager.add_pdfs(pdf_paths)
redis_manager = get_redis_manager()

print(redis_manager.get_count_hashes("processed_pdf_hashes"))

# Add endpoint and stuff..