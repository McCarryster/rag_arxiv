import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()


# Development or Production
PROD: bool = False


# API credentials
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Model configuration
TEXT_GENERATION_MODEL: str = "gpt-5-nano"
TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"


# Vector search setup
TOP_K: int = 15


# Paths
LOCAL_FAISS_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/vector_db"
LOCAL_BM25_PATH: str = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_arxiv/data/bm25_storage"