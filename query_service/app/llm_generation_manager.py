from langfuse import Langfuse
from typing import List, Optional, Dict
from langfuse.langchain import CallbackHandler
from langchain_core.documents import Document

from openai import OpenAI
import tiktoken

import prompt
import config

from fastapi.concurrency import run_in_threadpool

from utils import format_context_for_prompt


class LLMGenerationManager:
    def __init__(
        self,
        client: OpenAI,
        langfuse_tracing: Optional[Langfuse] = None
    ) -> None:
        self.client: OpenAI = client
        self.langfuse_tracing: Optional[Langfuse] = langfuse_tracing
        self.langfuse_handler = CallbackHandler() if langfuse_tracing else None

    def _generate_answer_sync(self, query: str, retrieved_docs) -> str:
        if self.langfuse_tracing:
            with self.langfuse_tracing.start_as_current_observation(
                as_type="generation",
                name="openai_answer_generation",
                model=config.TEXT_GENERATION_MODEL
            ) as generation_obs:
                context: str = format_context_for_prompt(retrieved_docs)
                
                # For Langfuse tracing - capture what you're sending
                trace_input = [
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"Question: {query}\n\n"
                            "Answer:"
                        ),
                    }
                ]
                
                generation_obs.update(input=trace_input)
                
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
                usage_details: Optional[Dict[str, int]] = None
                if resp.usage is not None:
                    usage_details = {
                        "input": resp.usage.input_tokens,
                        "output": resp.usage.output_tokens,
                        "total": resp.usage.total_tokens,
                    }
                generation_obs.update(
                    output=answer,
                    usage_details=usage_details,
                )
                return answer
        else:
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
            return resp.output_text.strip()

    async def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        return await run_in_threadpool(self._generate_answer_sync, query, retrieved_docs)