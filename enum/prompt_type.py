
from typing import Literal
from enum import Enum

class PromptType(str, Enum):
    REWRITE = "rewrite"
    CRITIC = "critic"
    DRAFTER = "drafter"
    PLANNER = "planner"
    HALLUCINATION_CHECKER = "hallucination_checker"

    




