
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from typing import Optional



@dataclass
class ToolSpec:
  name: str
  tool: BaseTool
  prompt_template : ChatPromptTemplate
  build_human_vars : Callable[[Any, str], Dict[str,Any]]      
  inject_args : Callable[[Dict[str, Any], Any], Dict[str, Any]] = lambda call_args, state: call_args 
  requires: List[str] = field(default_factory=list)    # name of tools required        
  max_attempts: int = 2
  fallback: Optional[str] = None        

