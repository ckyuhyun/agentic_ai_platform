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

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.messages import SystemMessage

PromptType = Literal["critic", "drafter", "planner"]

@dataclass
class PromptVersion:
    prompt_type: str
    version_id: str
    content: str
    description: str               = ""
    created_at: datetime           = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime           = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str]                = field(default_factory=list)

    def as_messages(self) -> List[SystemMessage]:
        """Return as a LangChain message list ready to prepend to a prompt."""
        return [SystemMessage(content=self.content)]

    def __repr__(self) -> str:
        return (
            f"PromptVersion(type={self.prompt_type!r}, version={self.version_id!r}, "
            f"created={self.created_at.strftime('%Y-%m-%d')}, tags={self.tags})"
        )


class PromptRegistry:
    def __init__(self):
        self.prompts_storage: Dict[str, Dict[str, PromptVersion]] = {}

    def register(self,
                 prompt_type: PromptType,
                 version_id: str,
                 content: str,
                 description: str = "",
                 tags: Optional[List[str]]= []) -> None:

        try:
            prompt_version = PromptVersion(
                prompt_type=prompt_type,
                version_id=version_id,
                content=content,
                description=description,
                tags=tags,
            )
            self.prompts_storage.setdefault(prompt_type, {})[version_id] = prompt_version
        except:
            raise ValueError(f"Failed to register prompt version: {prompt_type} {version_id}")
            
        
        

    def get_prompts_by_type_version_tags(self,
                    prompt_type: PromptType,
                    version_id: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> List[PromptVersion]:

        prompts = [p for p in self.prompts if p.prompt_type == prompt_type]

        if version_id is not None:
            prompts = [p for p in prompts if p.version_id == version_id]

        if tags is not None:
            for t in tags:
                prompts = [p for p in prompts if t in p.tags]

        return prompts
    
 
    def update(self,
               prompt_type: PromptType,
               version_id: str,
               content: str,
               description: str = "") -> None:
        
        try:
            if prompt_type in self.prompts_storage and version_id in self.prompts_storage[prompt_type]:
                self.prompts_storage[prompt_type][version_id].content = content
                self.prompts_storage[prompt_type][version_id].description = description
                self.prompts_storage[prompt_type][version_id].updated_at = datetime.now(timezone.utc)
            else:
                self.register(prompt_type, version_id, content, description)

        except:
            raise ValueError(f"Failed to update prompt version: {prompt_type} {version_id}")
        
    
        
    

        
        
        



