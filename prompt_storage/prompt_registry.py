import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Literal, Optional, Tuple, get_args

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langsmith import Client
from langsmith.prompt_cache import configure_global_prompt_cache
from langsmith.utils import LangSmithConflictError

from agentic_ai_platform.enum.prompt_type import PromptType
from agentic_ai_platform import logger



load_dotenv(Path(__file__).resolve().parents[2] / ".env")

@dataclass
class SystemPromptVersion:
    prompt_type: PromptType
    version_id: str
    system_prompt: str
    description: str               = ""
    created_at: datetime           = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime           = field(default_factory=lambda: datetime.now(timezone.utc))
    tag: str                = field(default_factory=str)

    def as_system_messages(self) -> List[SystemMessage]:
        """Return as a LangChain message list ready to prepend to a prompt."""
        return [SystemMessage(content=self.system_prompt)]
    
    def as_string_message(self) -> str:
        return self.system_prompt

    def __repr__(self) -> str:
        return (
            f"PromptVersion(type={self.prompt_type!r}, version={self.version_id!r}, "
            f"created={self.created_at.strftime('%Y-%m-%d')}, tag={self.tag})"
        )


class PromptRegistry:
    """
    PromptRegistry — versioned prompt management.

    Prompts are registered under a (prompt_type, version) key.
    The registry tracks which version is active per type and lets you
    compare, promote, and roll back versions at runtime.

    Usage:
        registry = PromptRegistry()

        # register versions
        registry.register("critic", "v1", "You are a strict critic...")
        registry.register("critic", "v2", "You are a strict but fair critic. Be concise.")

        # set active version
        registry.set_active("critic", "v2")

        # retrieve active prompt
        prompt = registry.get("critic")          # → PromptVersion for v2
        messages = prompt.as_messages()          # → [SystemMessage(...)]

        # compare two versions
        registry.diff("critic", "v1", "v2")
    """

    def __init__(self):

        self.is_register_to_langsmith = os.getenv("LANGSMITH_Prompt_Update").lower() == "true"
        self.prompts_storage: Dict[str, List[SystemPromptVersion]] = {}        
        self.prompt_id : str
        self.register_func: Callable[[PromptType, str, str, str, Optional[List[str]]], None] = None
        self.getprompt_func : Callable[[PromptType, str | None,  str | None], SystemPromptVersion] = None
        self.smith_client = Client()

    def register(self,
                 prompt_type: PromptType,
                 version_id: str,
                 promptTempate : ChatPromptTemplate ,
                 description: str | None = "",
                 tag: str | None = "") -> None:

        
        if self.is_register_to_langsmith:
            self.register_func = self.__register_to_langsmith__
            
        else:
            self.register_func = self.__register_to_local__
            
        system_prompt : str = next( (m for m in promptTempate.messages if isinstance(m, SystemMessagePromptTemplate)), 
                             None
                            ).prompt.template
            
        self.register_func(prompt_type=prompt_type,
                                version_id=version_id,
                                prompt=system_prompt,
                                description=description,
                                tag=tag)
        

    def register_with_str_prompt(self,
                 prompt_type: PromptType,
                 version_id: str,
                 prompt: str,
                 description: str | None = "",
                 tag: str | None = "") -> None:
        """
        register the prompt either to local or langsmith 
        """
        if self.is_register_to_langsmith:
            self.register_func = self.__register_to_langsmith__
        else:
            self.register_func = self.__register_to_local__
            
        self.register_func(prompt_type=prompt_type,
                            version_id=version_id,
                            prompt=prompt,
                            description=description,
                            tag=tag)

    def get_prompt(self,
                   prompt_type: PromptType,
                   version_id: str | None  = "",
                   tag: str | None = "") -> SystemPromptVersion:
        if self.is_register_to_langsmith:
            self.getprompt_func = self.__get_prompt_from_langsmith__
        else:
            self.getprompt_func = self.__get_prompt_by_type_version_tags__

        return self.getprompt_func(prompt_type, 
                                   version_id,
                                   tag)


    def __register_to_langsmith__(self,
                              prompt_type: PromptType,
                              version_id: str,
                            prompt: str,
                            description: str | None = "",
                            tag: str | None = "") -> None:
        """
        parameters:
        - prompt : only system prompt allowed
        """

        prompt_id = self.__get_prompt_id__()

        #if not ls._prompt_exists(prompt_id):
        try:
            prompt_template = ChatPromptTemplate.from_messages([("system" , prompt)])
            result = self.smith_client.push_prompt(prompt_id, object = prompt_template)
        except LangSmithConflictError as e:
            logger.info("This prompt is already the latest prompt")
        
        

    def __register_to_local__(self,
                 prompt_type: PromptType,
                 version_id: str,
                 prompt: str,
                 description: str | None = "",
                 tag: str | None = "") -> None:

        try:
            prompt_version = SystemPromptVersion(
                prompt_type=prompt_type,
                version_id=version_id,
                system_prompt=prompt,
                description=description,
                tag=tag,
            )
            self.prompts_storage.setdefault(prompt_type, []).append(prompt_version)
        except:
            raise ValueError(f"Failed to register prompt version: {prompt_type} {version_id}")
            
        
    def __get_prompt_from_langsmith__(self,
                                      prompt_type: PromptType,
                                      version_id: str | None = "",
                                      tag: str | None = "") -> SystemPromptVersion:
        fetched_prompt = self.smith_client.pull_prompt(self.__get_prompt_id__())
        system_prompt = next(
            (m for m in fetched_prompt.messages if isinstance(m, SystemMessagePromptTemplate)), 
            None
        ).prompt.template
        return SystemPromptVersion(
            prompt_type=prompt_type,
            version_id=version_id,
            tag=tag,
            system_prompt=system_prompt
        )

    def __get_prompt_by_type_version_tags__(self,
                    prompt_type: PromptType,
                    version_id: str | None = "",
                    tag: str | None = "") -> SystemPromptVersion:

        prompt_type_value = self.prompts_storage.get(prompt_type, {})
        

        if version_id is not None:
            prompt_version_value = [d for d in prompt_type_value if d.version_id == version_id]
            if len(prompt_version_value) == 0:
                raise ValueError(f"No prompt found for type {prompt_type} with version {version_id}")
            
            if len(prompt_version_value) > 1:
                prompt_version_value = [d for d in prompt_version_value if tag == d.tag ]

                if len(prompt_version_value) == 0:
                    raise ValueError(f"No prompt found for type {prompt_type} with version :{version_id} and tag :{tag}")

                if len(prompt_version_value) > 1:
                    raise ValueError(f"Multiple prompts found for type {prompt_type} with version {version_id}")
        else:
            pass

        # if tags is not None:
        #     for t in tags:
        #         prompts = [p for p in prompts if t in p.tags]

        return prompt_version_value[0]
    
 
    def update(self,
               prompt_type: PromptType,
               version_id: str,
               content: str,
               description: str = "") -> None:
        
        try:
            if prompt_type in self.prompts_storage and version_id in self.prompts_storage[prompt_type]:
                self.prompts_storage[prompt_type][version_id].system_prompt = content
                self.prompts_storage[prompt_type][version_id].description = description
                self.prompts_storage[prompt_type][version_id].updated_at = datetime.now(timezone.utc)
            else:
                self.__register_to_local__(prompt_type, version_id, content, description)

        except:
            raise ValueError(f"Failed to update prompt version: {prompt_type} {version_id}")
        
    
        
    def __get_prompt_id__(self):
        return "planner_track_system"
    

        
        
        



