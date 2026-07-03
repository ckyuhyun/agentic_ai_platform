from enum import Enum

class PromptType(str, Enum):
    REWRITE = "rewrite"
    REWRITE_EVAL = "rewrite_eval"
    EXECUTE_TOOLS = "execute_tools"
    CRITIC = "critic"
    DRAFTER = "drafter"
    PLANNER = "planner"
    HALLUCINATION_CHECKER = "hallucination_checker"

    




