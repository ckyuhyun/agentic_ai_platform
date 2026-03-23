import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

from agentic_ai_platform.graph.embedded_model_decision import OLLAMA_BASE_URL



class llm:
    def __init__(self, model_name: str):
        load_dotenv()

        self.model_name = model_name
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        self.model = self._llm_model_init_()

    
    
    def chat(self, 
             messages) -> str:
        response =self.model.invoke(messages)
        return response
    
    def _llm_model_init_(self):
        _ollama_models = {"llama3", "llama3.1", "llama3.2", "mistral", "gemma", "phi3", "qwen2"}

        model = None
        if self.model_name in _ollama_models or ":" in self.model_name and not self.model_name.startswith("gpt"):
            model = ChatOllama(
                model="llama3:latest",
                base_url=self.OLLAMA_BASE_URL,
                num_ctx=8192,
                temperature=0,
            )
        else:
            model = init_chat_model(
                model=self.model_name,
                temperature=0,
            )
        return model
        
