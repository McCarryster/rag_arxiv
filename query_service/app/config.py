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


# Langfuse set up
LANGFUSE_AVAILABLE = True
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "")


# Vector search setup
TOP_K: int = 15
VECTOR_DB_HYBRID_SEARCH_URL: str = os.getenv("VECTOR_DB_HYBRID_SEARCH_URL", "")
VECTOR_DB_METADATA_SEARCH_URL: str = os.getenv("VECTOR_DB_METADATA_SEARCH_URL", "")
# Text generation setup
TEMPERATURE: int = 0 # 0 for RAG factual accuracy

# Redis db config
RD = {
    "host": os.getenv("REDIS_HOST", ""),
    "port": os.getenv("REDIS_PORT", "6379"),
    "db": os.getenv("REDIS_DB", "0"),
    "password": os.getenv("REDIS_PASSWORD", None),
}