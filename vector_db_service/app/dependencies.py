from typing import Optional
import config
from vector_db_manager import WeaviateDBManager

_vector_db_manager: Optional[WeaviateDBManager] = None

def init_vector_db_manager() -> None:
    global _vector_db_manager

    if _vector_db_manager is None:
        _vector_db_manager = WeaviateDBManager(
            collection_name=config.COLLECTION_NAME,
            index_name=config.INDEX_NAME,
        )


def close_vector_db_manager() -> None:
    global _vector_db_manager

    if _vector_db_manager is not None:
        _vector_db_manager.close()
        _vector_db_manager = None


def get_vector_db_manager() -> WeaviateDBManager:
    if _vector_db_manager is None:
        raise RuntimeError("VectorDBManager not initialized")
    return _vector_db_manager