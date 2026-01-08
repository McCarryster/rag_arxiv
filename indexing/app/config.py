import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# API credentials
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model configuration
TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"
TEXT_GENERATION_MODEL: str = "gpt-5-nano"

# Vectorstore set up
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 15

# Redis db config
RD = {
    "host": os.getenv("REDIS_HOST", ""),
    "port": os.getenv("REDIS_PORT", 0),
    "db": os.getenv("REDIS_DB", 0),
    "password": os.getenv("REDIS_PASSWORD", ""),
}

# Development or Production
PROD: bool = False