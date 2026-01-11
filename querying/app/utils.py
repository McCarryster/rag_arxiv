from langchain_core.documents import Document
from typing import List


def format_context_for_prompt(docs: List[Document]) -> str:
    """
    Builds a string context where each chunk is labeled for the LLM to cite.
    Matches the requirement: [Paper ID:chunk]
    """
    context_parts: List[str] = []
    
    for doc in docs:
        paper_id: str = doc.metadata.get("source", "Unknown")
        chunk_no: int = doc.metadata.get("chunk_id", 0)
        content: str = doc.page_content.strip()
        
        # Create a clearly delimited block for the LLM
        header: str = f"--- [Paper ID:{paper_id}:chunk:{chunk_no}] ---"
        context_parts.append(f"{header}\n{content}")
        
    return "\n\n".join(context_parts)