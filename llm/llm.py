import os
import logging
import httpx
import asyncio
from typing import Awaitable, Callable, List, Any
from openai import InternalServerError, LengthFinishReasonError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from langchain.chat_models import init_chat_model

from agentic_ai_platform import logger


# Request-level resilience defaults, overridable via env without a code change.
DEFAULT_LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120")) #previous was 30
DEFAULT_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
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


def _retry_if_retryable_llm_error(retry_state) -> bool:
    # is_retryable_llm_error may swap the model on the LLM instance, so the
    # predicate needs `self`, which tenacity exposes as the wrapped method's
    # first positional arg.
    exc = retry_state.outcome.exception()
    return exc is not None and retry_state.args[0].is_retryable_llm_error(exc)


class LLM:
    def __init__(self,
                 model_name: str, 
                 temperature:float = 0.7):
        self.model_name = model_name
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        # Default routes through the model-gateway's /base/ prefix (see
        # docker-compose.yml's model-gateway service) rather than hitting
        # vllm-engine-basemodel's published port directly, so callers don't
        # need to track per-engine ports. Point VLLM_BASE_URL at an explicit
        # /finetuned/v1 URL (e.g. evaluate_candidate.py's --candidate-url) to
        # evaluate the fine-tuned candidate instead.
        self.VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:11434/base/v1")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/base/v1")
        # Local vLLM runs unauthenticated ("EMPTY"); a remote GPU host (e.g. a
        # RunPod pod behind its public proxy) should be started with --api-key.
        self.VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
        self._vllm_headers = {"Authorization": f"Bearer {self.VLLM_API_KEY}"}
        # max_tokens (the completion budget) and the prompt share the same
        # context window, so both must be derived from the model's real
        # max_model_len -- a hardcoded guess here silently drifts out of sync
        # whenever the vLLM server is restarted with a different context size,
        # and if max_tokens ever gets set >= TOKEN_LIMIT there's zero room
        # left for the prompt and every request fails.
        self._using_vllm_fallback = False
        self._bound_tools = None
        self.TOKEN_LIMIT = self._resolve_token_limit()
        self.MAX_OUTPUT_TOKENS = max(256, self.TOKEN_LIMIT // 4)
        self._temperature = temperature
        
        self._llm_model_instance = self._llm_model_init_()
        self.Batch_size = 30
        self.fallback_model = os.getenv("VLLM_Model")

        
    async def is_retryable_llm_error(self, 
                               exc: BaseException) -> bool:
        """
        Define a predicate to filter retryable vs non-retryable LLM exceptions.

        A Google API / rate-limit failure swaps this instance over to the
        self-hosted vLLM model, so the retry that follows runs against it.
        """

        # exception list excluding failures
        #if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout)):
        #    return True
        if isinstance(exc, (GoogleAPIError, GoogleRateLimitError)):
            await self._switch_to_vllm_fallback()
            return True

        return type(exc).__name__ in _TRANSIENT_LLM_EXCEPTION_NAMES

    async def _switch_to_vllm_fallback(self) -> bool:
        """
        Replace self._llm_model_ with the vLLM model named by VLLM_Model.
        Returns False (leaving the current model untouched) when there is
        nothing to switch to or the switch itself fails.
        """
        fallback_model = os.getenv("VLLM_Model")
        if self._using_vllm_fallback or not fallback_model:
            return False

        try:
            # The completion budget must come from the fallback's own context
            # window, not the hosted model's default.
            self.TOKEN_LIMIT = self._fetch_vllm_max_model_len(fallback_model, self.TOKEN_LIMIT)
            self.MAX_OUTPUT_TOKENS = max(256, self.TOKEN_LIMIT // 4)
            self._llm_mode = self._build_vllm_model(fallback_model)            
            
        except Exception as e:
            logger.warning("LLM: failed to switch %s to vLLM fallback %s: %s",
                           self.model_name, fallback_model, e)
            return False

        self._using_vllm_fallback = True
        self._refresh_llm_instance()
        logger.warning("LLM: %s unavailable, switched to vLLM fallback %s",
                       self.model_name, fallback_model)

        await asyncio.sleep(1)
        return True

    def _refresh_llm_instance(self):
        """Point llm_instance at _llm_model_, re-applying any bound tools."""

        self._llm_model_instance =  self._llm_model_init_()
        if self._bound_tools is None:
            return

        tools, tool_required = self._bound_tools
        if tool_required:
            self._llm_model_instance = self._llm_model_instance.bind_tools(tools, tool_choice="required")
        else:
            self._llm_model_instance = self._llm_model_instance.bind_tools(tools)

    _llm_call_retry = retry(
        reraise=True,
        stop=stop_after_attempt(DEFAULT_LLM_CALL_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=1, max=20, exp_base=2, jitter=1),
        retry=_retry_if_retryable_llm_error,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if not isinstance(value, float):
            raise "temperature should be float type"

        if value > 1.0 or value <0.0:
            raise "A range of temperature is 0.0 ~ 1.0"
        
        self._temperature = value
        # reinitialize with temperature set update
        self._llm_model_instance = self._llm_model_init_()


    def _is_vllm_model(self) -> bool:
        """
        
        """        
        if self._using_vllm_fallback or self.model_name in os.getenv("VLLM_Model_Key"):
            return True
        else:
            return False
        
        # self_hosted_keys = {os.getenv("LLM_Model_Key"), os.getenv("VLLM_Model_Key")} - {None, ""}
        # return self.model_name in self_hosted_keys


    def _resolve_token_limit(self, default: int = 2048) -> int:
        """
        Ask the vLLM server for this model's real max_model_len instead of
        guessing. Only applies to the vLLM-backed path; other backends
        (OpenAI, Ollama, etc.) keep the fallback default, which is only used
        for the batch-vs-single routing heuristic there, not for max_tokens.
        """
        if not self._is_vllm_model():
            return default

        return self._fetch_vllm_max_model_len(self._get_llm_model_name(), default)


    def _fetch_vllm_max_model_len(self, llm_model: str, default: int) -> int:
        try:
            resp = httpx.get(f"{self._get_current_llm_url()}/models", headers=self._vllm_headers, timeout=5)
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                if entry.get("id") == llm_model and entry.get("max_model_len"):
                    return int(entry["max_model_len"])
        except Exception as e:
            logger.warning(
                "LLM: failed to resolve max_model_len from %s; falling back to %d: %s",
                self._get_current_llm_url(), default, e,
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
        # Remembered so a fallback model swap can re-bind the same tools.
        self._bound_tools = (tools, tool_required)
        self._refresh_llm_instance()




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
        # A lambda, not the ainvoke() coroutine itself: llm_instance is swapped
        # on fallback, so the retry has to re-read it and build a fresh call.
        return await self._llm_invoke(lambda: self._llm_model_instance.ainvoke(system_human_message,
                                                                        config=config))

    async def _llm_invoke(self,
                          invoke_func : Callable[[], Awaitable[Any]]):
        """
        Await invoke_func(); if a Google API / rate-limit error occurs, switch
        to the vLLM fallback model and call invoke_func() again.

        invoke_func must be a zero-arg factory returning a new awaitable each
        call -- a coroutine object can only be awaited once and can't be called.
        """
        try:
            return await self._invoke_with_gateway_backoff(invoke_func)
        except Exception as e:            
            await self._switch_to_vllm_fallback()
            return await self._invoke_with_gateway_backoff(invoke_func)

    async def _invoke_with_gateway_backoff(self,
                                           invoke_func : Callable[[], Awaitable[Any]]):
        """
        Retry invoke_func() on 5xx from an OpenAI-compatible backend -- e.g. a
        502 from the model-gateway while vLLM is cold-starting. The OpenAI
        client raises these as openai.InternalServerError, not google's
        BadGateway.
        """
        for attempt in range(1, DEFAULT_LLM_CALL_MAX_ATTEMPTS + 1):
            try:
                return await invoke_func()
            except LengthFinishReasonError as e:
                logger.error("")
            except InternalServerError as e:
                if attempt == DEFAULT_LLM_CALL_MAX_ATTEMPTS:
                    raise
                delay = min(20, 2 ** attempt)
                logger.warning("LLM: %s returned %s (attempt %d/%d), retrying in %ds",
                               self.model_name, e.status_code, attempt,
                               DEFAULT_LLM_CALL_MAX_ATTEMPTS, delay)
                await asyncio.sleep(delay)
        


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
        
        result = await self._llm_model_instance.abatch(prompts, config={"max_concurrency": 4})
        return result

    async def invoke_with_prompt_template(self, 
                                          prompt:ChatPromptTemplate):
        return await self._llm_invoke(
            lambda : self._llm_model_instance.ainvoke(prompt))
        
    async def invoke_with_structured_llm(self, 
                                         scehema:Any,
                                         prompt:ChatPromptTemplate):
        # Built inside the lambda so a fallback swap re-wraps the new llm_instance.
        system_prompt, human_prompt = self._decode_human_system_prompt_from_chatTemplate(prompt)
        prompt_token = await self.get_prompt_token_count(system_prompt) + await self.get_prompt_token_count(human_prompt)

        logger.info(f"Used Token : {prompt_token}")

        response = await self._llm_invoke(
            lambda: self._llm_model_instance.with_structured_output(schema=scehema).ainvoke(prompt))

        return response
        

            
        
    def _decode_human_system_prompt_from_chatTemplate(self,
                                                      prompt:ChatPromptTemplate) -> tuple[str,str]:
        system_prompt = next((msg.content for msg in prompt if msg.type =="system"), None)
        human_prompt = next((msg.content for msg in prompt if msg.type =="human"), None)

        return system_prompt, human_prompt


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

        response = await self._llm_model_instance.ainvoke(message,
                                           config=config)
        return response

    def _get_tokenize_url(self, llm_model: str) -> str:
        """
        The tokenize endpoint lives at a different path depending on how the
        model is served: gemma's server exposes it under /v1, the other
        OpenAI-compatible vLLM servers at the root.
        """
        base_url = self._get_current_llm_url().rstrip('/')
        if "gemma" in llm_model:
            return f"{base_url}/tokenize"
        return f"{base_url.removesuffix('/v1')}/tokenize"

    async def _tokenize(self,
                        prompt_text:str):
        _llm_model = self._get_llm_model_name()
        _tokenizer_url = self._get_tokenize_url(_llm_model)

        # Runpod could not support the /v1/tokenize so returns 404
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url=_tokenizer_url,
                json={"model": _llm_model, "prompt": prompt_text},
                headers=self._vllm_headers,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
                )
            resp.raise_for_status()
        return resp

    def _get_llm_model_name(self):
        if self._using_vllm_fallback:
            return os.getenv("VLLM_Model")
        else:
            return os.getenv("LLM_Model")

    async def get_prompt_token_count(self,
                      prompt: str) -> int:
        """
        Estimate the number of tokens in the given text.

        Uses the ~4-characters-per-token rule of thumb (OpenAI's own guidance
        for English text) instead of a model-specific tokenizer.
        """
        
        token_count = 0        
        prompt_text = prompt.content if hasattr(prompt, 'content') else prompt
        if not self._is_vllm_model():
            # No vLLM /tokenize endpoint for hosted providers
            token_count = max(1, len(prompt_text) // 4)
        else:
            try:
                resp = await self._tokenize(prompt_text=prompt_text)
                token_count += resp.json()["count"]
            except Exception as e:
                # Token counting only feeds the batch-vs-single heuristic, so an
                # unreachable/missing /tokenize falls back to the estimate rather
                # than switching the whole instance to another model. (i.e. it wouldn't return tokens with runpod as it is not supported)
                logger.warning("LLM: /tokenize failed (%s); estimating token count", e)
                token_count = max(1, len(prompt_text) // 4)

            # async with httpx.AsyncClient() as client:
            #     try:
            #         resp = await client.post(
            #             url=f"{self._get_current_llm_url().removesuffix('/v1')}/tokenize",
            #             json={"model": llm_model, "prompt": prompt_text},
            #             headers=self._vllm_headers,
            #             timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
            #             )
            #         resp.raise_for_status()

            #     except (httpx.HTTPError, httpx.ConnectError):
            #         self._switch_to_vllm_fallback()
            #         resp = await client.post(
            #                                 url=f"{self._get_current_llm_url().removesuffix('/v1')}/tokenize",
            #                                 json={"model": llm_model, "prompt": prompt_text},
            #                                 headers=self._vllm_headers,
            #                                 timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
            #                                 )


                # if resp.status_code == 200:
                #     token_count+= resp.json()["count"]
                # else:
                #     # if not getting the token count from the VLLM tokenizer well
                #     token_count+= max(1, len(prompt_text) // 4)   

        return token_count

    def _get_current_llm_url(self):
        if self._using_vllm_fallback:
            return self.VLLM_BASE_URL
        else:
            return self.LLM_BASE_URL

    
    def _build_vllm_model(self, llm_model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=llm_model_name,
            base_url=self._get_current_llm_url(),
            api_key=self.VLLM_API_KEY,
            max_tokens=self.MAX_OUTPUT_TOKENS, # Must leave room for the prompt within TOKEN_LIMIT -- never set this to the full context size
            temperature=self.temperature,
            timeout=DEFAULT_LLM_TIMEOUT_SECONDS, # Wait up for a response
            max_retries=DEFAULT_LLM_MAX_RETRIES, # Retry up on failure
        )

    def _llm_model_init_(self):
        _local_docker_llm_models = {"llama3", "llama3.1", "llama3.2", "mistral", "phi3"}

        _llm_model_instance = None
        
        llm_model_name = self._get_llm_model_name()

        if self._using_vllm_fallback or self._is_vllm_model():
            # keep the fallback across re-inits (e.g. the temperature setter)
            _llm_model_instance = self._build_vllm_model(llm_model_name=llm_model_name)        
        elif self.model_name in _local_docker_llm_models or \
            ":" in llm_model_name and not "gpt" in llm_model_name:
            _llm_model_instance = ChatOllama(
                model="llama3.1:latest",
                base_url=self.OLLAMA_BASE_URL,
                num_ctx=8192,
                temperature=self.temperature,
                # ChatOllama has no native max_retries; timeout is forwarded to
                # the underlying httpx client via client_kwargs.
                client_kwargs={"timeout": DEFAULT_LLM_TIMEOUT_SECONDS},
            )
        elif 'gpt' in llm_model_name:
            _llm_model_instance = ChatOpenAI(
                model = llm_model_name,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,# Wait up for a response
                max_retries=DEFAULT_LLM_MAX_RETRIES,# Retry up on failure
            )
        elif 'gemini' in llm_model_name or 'gemma' in llm_model_name:
            if "runpod" in os.getenv("LLM_BASE_URL", ""):
                _llm_model_instance = ChatOpenAI(
                                            model = "gemma", # runpod recongnizes only 'gemma'
                                            openai_api_base = os.getenv("LLM_BASE_URL"),
                                            timeout=DEFAULT_LLM_TIMEOUT_SECONDS,# Wait up for a response
                                            max_retries=DEFAULT_LLM_MAX_RETRIES,# Retry up on failure
                                        )
            else:
                _llm_model_instance = ChatGoogleGenerativeAI(
                                    model = llm_model_name,
                                    temperature = self.temperature,
                                    timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
                                    max_retries=DEFAULT_LLM_MAX_RETRIES
                                )
                
            
        else:
            logger.warning("llm model instance not being assigned")
            raise ("llm model instance not being assigned")

        return _llm_model_instance
        
