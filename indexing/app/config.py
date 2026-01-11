import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()


# Development or Production
PROD: bool = False


# API credentials
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Model configuration
TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"


# Vectorstore set up
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


# Redis db config
RD = {
    "host": os.getenv("REDIS_HOST", ""),
    "port": os.getenv("REDIS_PORT", 0),
    "db": os.getenv("REDIS_DB", 0),
    "password": os.getenv("REDIS_PASSWORD", ""),
}


# Paths
LOCAL_FAISS_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/vector_db"
LOCAL_BM25_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/bm25_storage"
LOCAL_PDF_STORAGE_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/app_arxiv_pdfs"