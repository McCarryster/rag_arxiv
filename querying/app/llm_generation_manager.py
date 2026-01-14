from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from typing import List, Optional
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from utils import format_context_for_prompt

class LLMGenerationManager:
    def __init__(
        self,
        text_generation_model: ChatOpenAI,
        prompt_template: ChatPromptTemplate,
        langfuse_tracing: Optional[Langfuse] = None
    ) -> None:
        self.text_generation_model: ChatOpenAI = text_generation_model
        self.prompt_template: ChatPromptTemplate = prompt_template
        self.langfuse_tracing: Optional[Langfuse] = langfuse_tracing
        self.langfuse_handler = CallbackHandler() if langfuse_tracing else None

    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        chain = (
            {
                "context": lambda x: format_context_for_prompt(retrieved_docs),
                "query": RunnablePassthrough()
            }
            | self.prompt_template
            | self.text_generation_model
            | StrOutputParser()
        )

        if not self.langfuse_handler:
            return chain.invoke(query)
        
        result = chain.invoke(
            query, 
            config={"callbacks": [self.langfuse_handler]}
        )
        
        if self.langfuse_tracing:
            self.langfuse_tracing.flush()
        
        return result