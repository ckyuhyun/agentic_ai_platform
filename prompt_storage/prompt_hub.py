from typing import Any, Literal
from langchain_core.messages import SystemMessage

class prompt_hub:
    
    def __init__(self, 
                 prompt_type: Literal['critic', 'plan']):
          self.prompt_type = prompt_type
        
    def get_system_prompt(self) -> Any:
        if self.prompt_type == 'critic':
            return [
                SystemMessage(content = 
            "You are a strict but fair critic. Evaluate the draft against the original task. "
            "Return your evaluation as structured data matching the requested schema exactly."
            "If hallucinations are suspected from draft, please provide detailed feedback to call hallucination_checker tool for further analysis.")
            ]
        elif self.prompt_type == 'plan':
            return [
                SystemMessage(content = 
            "You are a helpful assistant that creates a plan to solve the given task. "
            "Return your plan as structured data matching the requested schema exactly.")
            ]
        else:
            raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

        