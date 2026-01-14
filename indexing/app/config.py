import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()


# Development or Production
PROD: bool = False


# OpenAI API credentials
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Model configuration
TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Langfuse set up
LANGFUSE_AVAILABLE = True
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "")

# Vectorstore set up
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


# Redis db config
RD = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", None),
}


# Paths
LOCAL_FAISS_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/vector_db"
LOCAL_BM25_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/bm25_storage"
LOCAL_PDF_STORAGE_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/app_arxiv_pdfs"