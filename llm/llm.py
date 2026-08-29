import os
import httpx
import asyncio
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
        self.model_name = model_name
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        # Default routes through the model-gateway's /base/ prefix (see
        # docker-compose.yml's model-gateway service) rather than hitting
        # vllm-engine-basemodel's published port directly, so callers don't
        # need to track per-engine ports. Point VLLM_BASE_URL at an explicit
        # /finetuned/v1 URL (e.g. evaluate_candidate.py's --candidate-url) to
        # evaluate the fine-tuned candidate instead.
        self.VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:11434/base/v1")
        # max_tokens (the completion budget) and the prompt share the same
        # context window, so both must be derived from the model's real
        # max_model_len -- a hardcoded guess here silently drifts out of sync
        # whenever the vLLM server is restarted with a different context size,
        # and if max_tokens ever gets set >= TOKEN_LIMIT there's zero room
        # left for the prompt and every request fails.
        self.TOKEN_LIMIT = self._resolve_token_limit()
        self.MAX_OUTPUT_TOKENS = max(256, self.TOKEN_LIMIT // 4)
        self._llm_model_ = self._llm_model_init_()
        self.llm_instance = self._llm_model_
        self.Batch_size = 30

    def _resolve_token_limit(self, default: int = 2048) -> int:
        """
        Ask the vLLM server for this model's real max_model_len instead of
        guessing. Only applies to the vLLM-backed path; other backends
        (OpenAI, Ollama, etc.) keep the fallback default, which is only used
        for the batch-vs-single routing heuristic there, not for max_tokens.
        """
        llm_model_key = os.getenv("LLM_Model_Key") or ""
        if self.model_name not in llm_model_key:
            return default

        llm_model = os.getenv("LLM_Model")
        try:
            resp = httpx.get(f"{self.VLLM_BASE_URL}/models", timeout=5)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == llm_model and entry.get("max_model_len"):
                    return int(entry["max_model_len"])
        except Exception as e:
            logger.warning(
                "LLM: failed to resolve max_model_len from %s; falling back to %d: %s",
                self.VLLM_BASE_URL, default, e,
            )
        return default

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
        prompt_token_count =  await self.get_prompt_token_count(system_human_message)
        

        

        # Reserve room for the completion: TOKEN_LIMIT is the total context
        # window (prompt + output combined), so route to batching once the
        # prompt alone would leave too little space for MAX_OUTPUT_TOKENS.
        input_budget = self.TOKEN_LIMIT - self.MAX_OUTPUT_TOKENS
        if prompt_token_count >= input_budget:

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




    
    async def invoke(self, 
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

        response = await self.llm_instance.ainvoke(message,
                                           config=config)
        return response

    async def get_prompt_token_count(self,
                      prompts: List[Any]) -> int:
        """
        Estimate the number of tokens in the given text.

        Uses the ~4-characters-per-token rule of thumb (OpenAI's own guidance
        for English text) instead of a model-specific tokenizer.
        """
        llm_model = os.getenv("LLM_Model")
        token_count = 0
        for prompt in prompts:
            prompt_text = prompt.content
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.VLLM_BASE_URL.removesuffix('/v1')}/tokenize",
                    json={"model": llm_model, "prompt": prompt_text},
                    timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
                    )
                resp.raise_for_status()

                if resp.status_code == 200:
                    token_count+= resp.json()["count"]
                else:
                    # if not getting the token count from the VLLM tokenizer well
                    token_count+= max(1, len(prompt_text) // 4)   

        return token_count
    
    def _llm_model_init_(self):
        _local_docker_llm_models = {"llama3", "llama3.1", "llama3.2", "mistral", "gemma", "phi3"}

        model = None
        llm_model_key = os.getenv("LLM_Model_Key")        
        llm_model = os.getenv("LLM_Model")

        if self.model_name in llm_model_key:
            model = ChatOpenAI(
                model=llm_model,
                base_url=self.VLLM_BASE_URL,
                api_key="EMPTY",
                max_tokens=self.MAX_OUTPUT_TOKENS, # Must leave room for the prompt within TOKEN_LIMIT -- never set this to the full context size
                temperature=0.7,
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
                temperature=0.7,
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
                temperature=0.7,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,# Wait up for a response
                max_retries=DEFAULT_LLM_MAX_RETRIES,# Retry up on failure
            )
        return model
        
