from typing import List
from agentic_ai_platform.enum.prompt_type import PromptType


class PromptSpec:
     prompt_template_version : str | None
     prompt_template_tags : List[str] | None
     prompt_template_type : PromptType | None