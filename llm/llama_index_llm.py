import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from agentic_ai_platform.llm.llm import LLM


class LlamaIndexLLM(CustomLLM):
    """
    Exposes the platform LLM (primary model + vLLM fallback) to LlamaIndex,
    e.g. as the response synthesizer of a query engine.
    """

    model_name: str
    # Sized for the vLLM fallback (the smaller of the two backends) so a prompt
    # LlamaIndex packs for the primary model still fits after a fallback swap.
    context_window: int = 4096
    num_output: int = 1024

    _llm: LLM = PrivateAttr()

    def __init__(self,
                 model_name: str,
                 temperature: float = 0.7,
                 **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self._llm = LLM(model_name=model_name, temperature=temperature)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    async def _ainvoke(self, prompt: str) -> CompletionResponse:
        response = await self._llm.invoke_by_single_prompt([HumanMessage(content=prompt)])
        if response is None:
            raise RuntimeError(f"LLM {self.model_name} returned no response")

        content = response.content
        return CompletionResponse(text=content if isinstance(content, str) else response.text)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return await self._ainvoke(prompt)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        # LLM is async-only; a worker thread gets its own event loop, so this
        # also works when the caller is already inside a running loop.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._ainvoke(prompt)).result()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        # LLM has no streaming call, so the whole answer arrives as one chunk.
        response = self.complete(prompt, formatted=formatted, **kwargs)
        yield CompletionResponse(text=response.text, delta=response.text)
