from langfuse import Langfuse
from typing import List, Optional, Dict, Any
from langfuse.langchain import CallbackHandler
from langchain_core.documents import Document

from openai import OpenAI
import tiktoken
from langfuse_decorator import async_trace

import prompt
import config

from fastapi.concurrency import run_in_threadpool

from utils import format_context_for_prompt


class LLMGenerationManager:
    def __init__(self, client: OpenAI) -> None:
        self.client: OpenAI = client

    # def _generate_answer_sync(self, query: str, retrieved_docs) -> str:
    #     if self.langfuse_tracing:
    #         with self.langfuse_tracing.start_as_current_observation(
    #             as_type="generation",
    #             name="openai_answer_generation",
    #             model=config.TEXT_GENERATION_MODEL
    #         ) as generation_obs:
    #             context: str = format_context_for_prompt(retrieved_docs)
                
    #             # For Langfuse tracing - capture what you're sending
    #             trace_input = [
    #                 {
    #                     "role": "user",
    #                     "content": (
    #                         f"Context:\n{context}\n\n"
    #                         f"Question: {query}\n\n"
    #                         "Answer:"
    #                     ),
    #                 }
    #             ]
                
    #             generation_obs.update(input=trace_input)
                
    #             resp = self.client.responses.create(
    #                 model=config.TEXT_GENERATION_MODEL,
    #                 instructions=prompt.SYSTEM_PROMPT,
    #                 input=[
    #                     {
    #                         "role": "user",
    #                         "content": (
    #                             f"Context:\n{context}\n\n"
    #                             f"Question: {query}\n\n"
    #                             "Answer:"
    #                         ),
    #                     }
    #                 ],
    #                 store=False,
    #             )
                
    #             answer = resp.output_text.strip()
    #             usage_details: Optional[Dict[str, int]] = None
    #             if resp.usage is not None:
    #                 usage_details = {
    #                     "input": resp.usage.input_tokens,
    #                     "output": resp.usage.output_tokens,
    #                     "total": resp.usage.total_tokens,
    #                 }
    #             generation_obs.update(
    #                 output=answer,
    #                 usage_details=usage_details,
    #             )
    #             return answer
    #     else:
    #         context: str = format_context_for_prompt(retrieved_docs)
    #         resp = self.client.responses.create(
    #             model=config.TEXT_GENERATION_MODEL,
    #             instructions=prompt.SYSTEM_PROMPT,
    #             input=[
    #                 {
    #                     "role": "user",
    #                     "content": (
    #                         f"Context:\n{context}\n\n"
    #                         f"Question: {query}\n\n"
    #                         "Answer:"
    #                     ),
    #                 }
    #             ],
    #             store=False,
    #         )
    #         return resp.output_text.strip()

    @async_trace(name="query pipeline", model=config.TEXT_GENERATION_MODEL)
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