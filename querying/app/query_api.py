import config
from typing import List, Dict, Any
from pydantic import BaseModel, SecretStr
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dependencies import get_vector_search_manager
from utils import format_context_for_prompt
import prompt


# --- Models ---
class QueryRequest(BaseModel):
    user_query: str

class IndexResponse(BaseModel):
    message: str
    model_response: str
    sources: List[Dict[str, Any]]


app = FastAPI(title="Arxiv Query Service")


# --- Endpoint ---
@app.post("/query", response_model=IndexResponse)
async def query_index(request: QueryRequest) -> IndexResponse:
    """
    Performs hybrid search and generates a response based on retrieved Arxiv context.
    """
    # 1. Setup Manager
    vector_search_manager = get_vector_search_manager()

    # 2. Model Setup (temperature=0 for factual accuracy for RAG)
    text_generation_model = ChatOpenAI(
        api_key=SecretStr(config.OPENAI_API_KEY), 
        model=config.TEXT_GENERATION_MODEL,
        temperature=0
    )
    
    # 3. Retrieval
    retrieved_docs: List[Document] = vector_search_manager.perform_hybrid_search(request.user_query)
    if not retrieved_docs:
        return IndexResponse(
            message="No matches",
            model_response="Insufficient relevant details in retrieved papers.",
            sources=[]
        )

    # 4. Create the ChatPromptTemplate. We use a list of tuples (role, content) for modern Chat Models
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt.SYSTEM_PROMPT),
        ("human", "{query}")
    ])

    # 5. Build the LCEL Chain. map 'context' and 'query' keys to match your SYSTEM_PROMPT placeholders
    chain = (
        {
            "context": lambda x: format_context_for_prompt(retrieved_docs),
            "query": RunnablePassthrough()
        }
        | prompt_template
        | text_generation_model
        | StrOutputParser()
    )

    # 6. Execute
    answer: str = await chain.ainvoke(request.user_query)

    # 7. Collect metadata for frontend display
    sources = [
        {
            "id": d.metadata.get("source"),
            "chunk": d.metadata.get("chunk_id"),
            "hash": d.metadata.get("file_hash")
        } 
        for d in retrieved_docs
    ]

    return IndexResponse(
        message="Success",
        model_response=answer,
        sources=sources
    )