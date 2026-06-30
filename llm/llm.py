import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


# Opt-in alias -> actual model id served by the local vllm-engine container
# (see agentic_ai_platform/docker-compose.yml). Pass model_name="qwen2.5-local"
# to route through it instead of Claude/Ollama.
_VLLM_MODELS = {
    "qwen2.5-local": "Qwen/Qwen2.5-3B-Instruct-AWQ",
}



class LLM:
    def __init__(self, 
                 model_name: str):
        load_dotenv()

        self.model_name = model_name
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        self.VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._llm_model_ = self._llm_model_init_()
        self.llm_instance = self._llm_model_

    def bind_tools(self,
                   tools: list,
                   tool_choice: str = None):
        """Bind tools to the LLM so it can call them during inference.

        tool_choice="required" forces the model to emit a tool call instead of
        a freeform text response (supported by OpenAI-compatible endpoints,
        including vLLM with --enable-auto-tool-choice).
        """
        if tool_choice is not None:
            self.llm_instance = self._llm_model_.bind_tools(tools, tool_choice=tool_choice)
        else:
            self.llm_instance = self._llm_model_.bind_tools(tools)


    def invoke_by_single_prompt(self,
                                system_human_message:str) -> str:
        """
        Invoke the LLM with the given system and human messages, and return the response.
        """
        
        response =self.llm_instance.invoke(system_human_message)
        return response    
    
    def invoke(self, 
             system_message: str = None,
             human_message: str = None) -> str:
        """
        Invoke the LLM with the given system and human messages, and return the response.
        """
        message = []

        if system_message:
            message.append(SystemMessage(content=system_message))
        if human_message:
            message.append(HumanMessage(content=human_message))

        response =self.llm_instance.invoke(message)
        return response
    
    def _llm_model_init_(self):
        _local_docker_llm_models = {"llama3", "llama3.1", "llama3.2", "mistral", "gemma", "phi3", "qwen2"}

        model = None
        if self.model_name in _VLLM_MODELS:
            model = ChatOpenAI(
                model=_VLLM_MODELS[self.model_name],
                base_url=self.VLLM_BASE_URL,
                api_key="EMPTY",
                temperature=0,
            )
        elif self.model_name in _local_docker_llm_models or \
            ":" in self.model_name and not \
                self.model_name.startswith("gpt"):
            model = ChatOllama(
                model="llama3.1:latest",
                base_url=self.OLLAMA_BASE_URL,
                num_ctx=8192,
                temperature=0,
            )
        elif self.model_name.startswith('gpt'):
            model = ChatOpenAI(
                model = self.model_name
            )
        else:
            model = init_chat_model(
                model=self.model_name,
                temperature=0,
            )
        return model
        
