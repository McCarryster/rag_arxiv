from typing import List, Dict, Any
from langchain_core.documents import Document
import time

from openai import OpenAI
from langfuse_decorator import async_trace

import prompt
import config

from fastapi.concurrency import run_in_threadpool

from utils import format_context_for_prompt


class LLMGenerationManager:
    def __init__(self, client: OpenAI) -> None:
        self.client: OpenAI = client

    @async_trace(name="llm answer generation", model=config.TEXT_GENERATION_MODEL)
    async def generate_answer(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        return await run_in_threadpool(self._generate_answer_sync, query, retrieved_docs)

    def _generate_answer_sync(self, query: str, retrieved_docs) -> Dict[str, Any]:
        context: str = format_context_for_prompt(retrieved_docs)
        resp = self.client.responses.create(
            model=config.TEXT_GENERATION_MODEL,
            instructions=prompt.SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer:"
                    ),
                }
            ],
            max_output_tokens=256,              # hard cap (includes reasoning tokens)
            reasoning={"effort": "minimal"},    # or "low"
            text={"verbosity": "low"},          # less verbose wording
            store=False,
        )
        answer = resp.output_text.strip()

        input_tokens: int = 0
        output_tokens: int = 0
        total_tokens: int = 0
        if resp.usage is not None:
            input_tokens: int = resp.usage.input_tokens
            output_tokens: int = resp.usage.output_tokens
            total_tokens: int = resp.usage.total_tokens

        result: Dict[str, Any] = {
            "result": answer,
            "input": query,
            "output": answer,
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
            "metadata": {}
            }
        return result