import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()


COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "arxiv_papers_docs")
INDEX_NAME: str = os.getenv("INDEX_NAME", "arxiv_papers_index")


WEAVIATE_HTTP_HOST: str = os.getenv("WEAVIATE_HTTP_HOST", "weaviate")
WEAVIATE_HTTP_PORT: int = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))

WEAVIATE_GRPC_HOST: str = os.getenv("WEAVIATE_GRPC_HOST", "weaviate")
WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))