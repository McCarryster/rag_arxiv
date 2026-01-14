import pickle
from typing import List, Dict, Any, Protocol, Optional, runtime_checkable
from pathlib import Path
import tiktoken

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from langfuse import Langfuse

import config

# Protocols for providers
@runtime_checkable
class VectorStorageProvider(Protocol):
    def load(self, embedding_model: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]: ...
    def exists(self, index_name: str) -> bool: ...

@runtime_checkable
class BM25StorageProvider(Protocol):
    def load(self, index_name: str) -> Optional[BM25Retriever]: ...
    def exists(self, index_name: str) -> bool: ... # Added for consistency


# Storage providers
class LocalFaissStorageProvider:
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def load(self, embedding_model: OpenAIEmbeddings, index_name: str) -> Optional[FAISS]:
        load_path: Path = self.base_path / index_name
        if self.exists(index_name):
            return FAISS.load_local(
                str(load_path), 
                embedding_model,
                allow_dangerous_deserialization=True
            )
        return None

    def exists(self, index_name: str) -> bool:
        # FAISS indexes usually exist as a directory containing index.faiss
        return (self.base_path / index_name).exists()

class LocalBM25StorageProvider:
    """Handles loading and existence checks for the BM25 sparse index."""
    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def exists(self, index_name: str) -> bool:
        """Checks if the specific .pkl file exists on disk."""
        load_path: Path = self.base_path / f"{index_name}_bm25.pkl"
        return load_path.exists()

    def load(self, index_name: str) -> Optional[BM25Retriever]:
        """Loads the BM25 retriever if it exists."""
        load_path: Path = self.base_path / f"{index_name}_bm25.pkl"
        if self.exists(index_name):
            with open(load_path, "rb") as f:
                return pickle.load(f)
        return None


# Manager for vector search operations
class VectorSearchManager:
    def __init__(
        self,
        embedding_model: OpenAIEmbeddings,
        faiss_storage_provider: VectorStorageProvider,
        bm25_storage_provider: BM25StorageProvider,
        index_name: str,
        top_k: int = 4,
        faiss_retriever_weight: float = 0.5,
        bm25_retriever_weight: float = 0.5,
        langfuse_tracing: Optional[Langfuse] = None
    ) -> None:
        self.embedding_model: OpenAIEmbeddings = embedding_model
        self.faiss_storage: VectorStorageProvider = faiss_storage_provider
        self.bm25_storage: BM25StorageProvider = bm25_storage_provider
        self.index_name: str = index_name
        self.top_k: int = top_k
        self.faiss_retriever_weight: float = faiss_retriever_weight
        self.bm25_retriever_weight: float = bm25_retriever_weight
        self.langfuse_tracing: Optional[Langfuse] = langfuse_tracing

        # Load instances
        self.vector_store: Optional[FAISS] = self.faiss_storage.load(self.embedding_model, self.index_name)
        self.bm25_retriever: Optional[BM25Retriever] = self.bm25_storage.load(self.index_name)

        if not self.vector_store or not self.bm25_retriever:
            raise ValueError(f"Failed to load indexes for {index_name}")

        # Configure individual retrievers
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self.bm25_retriever.k = self.top_k

        # Initialize the Ensemble Retriever once during init for better performance
        self.ensemble_retriever: EnsembleRetriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_retriever_weight, self.faiss_retriever_weight]
        )

    def perform_hybrid_search(self, query: str) -> List[Document]:
        """
        Uses the pre-configured ensemble retriever to find documents.
        """
        return self.ensemble_retriever.invoke(query)

    def get_docs_by_metadata(self, metadata_list: List[Dict[str, Any]]) -> List[Document]:
            """
            Retrieves full Document objects from the docstore that match a list of metadata filters.
            
            Args:
                metadata_list: A list of dictionaries, where each dict contains metadata 
                            attributes (e.g., {'file_hash': '...', 'chunk_id': 1}).
            
            Returns:
                A list of Document objects matching the provided metadata sets.
            """
            if not self.vector_store or not isinstance(self.vector_store.docstore, InMemoryDocstore):
                return []

            matched_documents: List[Document] = []
            doc_dict: Dict[str, Document] = getattr(self.vector_store.docstore, "_dict", {})
            
            # Optimization: We iterate through the docstore once and check against all filters
            # if the docstore is large, or iterate through filters if the filter list is small.
            # Given cached_metadata is usually top_k (small), we iterate through filters.
            for criteria in metadata_list:
                for doc in doc_dict.values():
                    # Check if all keys in the criteria dict match the document metadata
                    match: bool = all(
                        doc.metadata.get(key) == value 
                        for key, value in criteria.items()
                    )
                    if match:
                        matched_documents.append(doc)
                        # Break to move to next criteria if we assume 1:1 match for chunk_id
                        break 

            return matched_documents

    def generate_embeddings(self, text: str) -> List[float]:
        if self.langfuse_tracing:
            trace_summary: Dict[str, Any] = {"total_character_length": len(text)}
            embedding_obs = self.langfuse_tracing.start_observation(
                name="openai_embedding_generation",
                as_type="embedding",
                model=config.TEXT_EMBEDDING_MODEL,
                input=trace_summary
            )
            try:
                embedding: List[float] = self.embedding_model.embed_query(text)
                # Calculate token usage
                encoding = tiktoken.encoding_for_model(config.TEXT_EMBEDDING_MODEL)
                total_input_tokens: int = len(encoding.encode(text))
                # Update observation with usage and metadata
                embedding_obs.update(
                    usage={
                        "input": total_input_tokens,
                        "total": total_input_tokens,
                    },
                    output={"dimensions": len(embedding)}  # log the metadata/dimensions
                )
                embedding_obs.end()
                self.langfuse_tracing.flush()
                return embedding
            except Exception as e:
                embedding_obs.update(level="ERROR", status_message=str(e))  # Ensure errors are logged to the trace if generation fails
                embedding_obs.end()
                raise e
        else:
            return self.embedding_model.embed_query(text)