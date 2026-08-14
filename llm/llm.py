import os
import httpx
import json
from dotenv import load_dotenv
from typing import List, Any
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from agentic_ai_platform import logger




# Opt-in alias -> actual model id served by the local vllm-engine container
# (see agentic_ai_platform/docker-compose.yml). Pass model_name="qwen2.5-local"
# to route through it instead of Claude/Ollama.
_VLLM_MODELS = {
    "qwen2.5-local": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Request-level resilience defaults, overridable via env without a code change.
DEFAULT_LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
DEFAULT_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Retries at the call-site (whole invoke(), not just the SDK's own transport
# retry) so backends with no native retry support (e.g. ChatOllama) still get
# backoff, and so a run outlives transient errors the SDK-level retry gave up
# on. Independent of DEFAULT_LLM_MAX_RETRIES above.
DEFAULT_LLM_CALL_MAX_ATTEMPTS = int(os.getenv("LLM_CALL_MAX_ATTEMPTS", "3"))

# Matched by exception class name rather than importing each provider SDK's
# (openai/anthropic/ollama) exception hierarchy directly, since LLM routes to
# whichever backend model_name resolves to and we don't want a hard import
# dependency on every one of them.
_TRANSIENT_LLM_EXCEPTION_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "TimeoutError",
    "ConnectionError",
}


def is_retryable_llm_error(exc: BaseException) -> bool:
    """
    Define a predicate to filter retryable vs non-retryable LLM exceptions
    """

    # exception list excluding failures
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout)):
        return True
    return type(exc).__name__ in _TRANSIENT_LLM_EXCEPTION_NAMES


_llm_call_retry = retry(
    reraise=True,
    stop=stop_after_attempt(DEFAULT_LLM_CALL_MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=1, max=20, exp_base=2, jitter=1),
    retry=retry_if_exception(is_retryable_llm_error),
    before_sleep=before_sleep_log(logger, logger.warning or 20),
)



class LLM:
    def __init__(self, 
                 model_name: str):
        load_dotenv()

        self.model_name = model_name
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        self.VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._llm_model_ = self._llm_model_init_()
        self.llm_instance = self._llm_model_
        self.TOKEN_LIMIT = 2048
        self.Batch_size = 30

    def bind_tools(self,
                   tools: list,
                   tool_required: bool = False):
        """Bind tools to the LLM so it can call them during inference.

        tool_choice="required" forces the model to emit a tool call instead of
        a freeform text response (supported by OpenAI-compatible endpoints,
        including vLLM with --enable-auto-tool-choice).
        """
        
        if tool_required:
            self.llm_instance = self._llm_model_.bind_tools(tools, tool_choice="required")
        else:
            self.llm_instance = self._llm_model_.bind_tools(tools)


    @_llm_call_retry
    async def invoke_by_single_prompt(self,
                                system_human_message:List[Any],
                                config : dict = None):
        """
        Invoke the LLM with the given system and human messages, and return the response.
        """
        system_prompt, human_prompt = self._decode_human_system_prompt(system_human_message)
        human_prompt_len = self.get_token_len(human_prompt)
        system_prompt_len = self.get_token_len(system_prompt)
        if ( human_prompt_len+ system_prompt_len)*2 >= self.TOKEN_LIMIT:
            
            response = await self._batch_invoke(system_message=system_prompt,
                                          human_message = human_prompt,
                                         config=config)
        else:
            response = await self._single_invoke(system_human_message=system_human_message,
                                            config=config)

        return response

       

    async def _single_invoke(self,
                       system_human_message:str,
                       config : dict = None) -> str:

        return await self.llm_instance.ainvoke(system_human_message,
                                        config=config)

    async def _batch_invoke(self,
                      system_message:str,
                      human_message:str,
                      config : dict = None) -> str | List[str]:

        # TODO: update how to chunk
        chunks = [human_message[i: i+self.Batch_size] for i in range(0, len(human_message), self.Batch_size)]

        prompts = []
        for idx, c in enumerate(chunks):
            prompt = ChatPromptTemplate.from_messages([("system", system_message),
                                                       ("human",c)])

            
            prompts.append(prompt.format_messages())

        result = await self.llm_instance.abatch(prompts, config={"max_concurrency": 4})
        return result
        

    def _decode_human_system_prompt(self,
                          system_human_message:List[Any]) -> tuple[str, str]:
        system_prompt  = ""
        human_prompt  = ""
        for msg in system_human_message:
            if isinstance(msg, HumanMessage):
                human_prompt = msg.content
            elif isinstance(msg, SystemMessage):
                system_prompt = msg.content
        
        return system_prompt, human_prompt




    
    def invoke(self, 
             system_message: str = None,
             human_message: str = None,
             config : dict = None) -> str:
        """
        Invoke the LLM with the given system and human messages, and return the response.
        """
        message = []

        if system_message:
            message.append(SystemMessage(content=system_message))
        if human_message:
            message.append(HumanMessage(content=human_message))

        response =self.llm_instance.ainvoke(message,
                                           config=config)
        return response

    def get_token_len(self,
                      text: str) -> int:
        """
        Estimate the number of tokens in the given text.

        Uses the ~4-characters-per-token rule of thumb (OpenAI's own guidance
        for English text) instead of a model-specific tokenizer, since local/
        Ollama/vLLM models (e.g. qwen2.5-local) have no tiktoken encoding.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def _llm_model_init_(self):
        _local_docker_llm_models = {"llama3", "llama3.1", "llama3.2", "mistral", "gemma", "phi3", "qwen2"}

        model = None
        if self.model_name in _VLLM_MODELS:
            model = ChatOpenAI(
                model=_VLLM_MODELS[self.model_name],
                base_url=self.VLLM_BASE_URL,
                api_key="EMPTY",
                temperature=0,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS, # Wait up for a response
                max_retries=DEFAULT_LLM_MAX_RETRIES, # Retry up on failure
            )
        elif self.model_name in _local_docker_llm_models or \
            ":" in self.model_name and not \
                self.model_name.startswith("gpt"):
            model = ChatOllama(
                model="llama3.1:latest",
                base_url=self.OLLAMA_BASE_URL,
                num_ctx=8192,
                temperature=0,
                # ChatOllama has no native max_retries; timeout is forwarded to
                # the underlying httpx client via client_kwargs.
                client_kwargs={"timeout": DEFAULT_LLM_TIMEOUT_SECONDS},
            )
        elif self.model_name.startswith('gpt'):
            model = ChatOpenAI(
                model = self.model_name,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,# Wait up for a response
                max_retries=DEFAULT_LLM_MAX_RETRIES,# Retry up on failure
            )
        else:
            model = init_chat_model(
                model=self.model_name,
                temperature=0,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,# Wait up for a response
                max_retries=DEFAULT_LLM_MAX_RETRIES,# Retry up on failure
            )
        return model
        
