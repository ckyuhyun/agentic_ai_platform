
from typing import List, Type
from pydantic import BaseModel, Field

class HallucinationSignal(BaseModel):
    pattern: str = Field("", description="The specific hallucination pattern detected")
    severity: str = Field("", description="'WARN' for concerning but not critical, 'FAIL' for high-risk hallucinations")
    excerpt: str = Field("", description="The exact text snippet from the agent's output that triggered the signal")
    HallucinationSignal_logs : List[str] = Field(default_factory=list, description="Detailed logs of the hallucination check process for debugging and analysis")

class SafetyJudge(BaseModel):
        score : str = Field("", description="The safety score of the response, e.g., 'safe', 'unsafe', 'uncertain'")
        HallucinationSignal_result :HallucinationSignal = Field(description="Detailed results from the hallucination check, including any detected signals")


class HallucinationCheckerConfig(BaseModel):
    hallucination_system_prompt_version_id : str = Field(description="Version ID for the system prompt to use in the hallucination checker", default="0")
