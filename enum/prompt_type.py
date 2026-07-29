from enum import Enum

class PromptType(str, Enum):
    REWRITE = "rewrite"
    REWRITE_EVAL = "rewrite_eval"
    EXECUTE_TOOLS = "execute_tools"
    CRITIC = "critic"
    DRAFTER = "drafter"
    PLANNER = "planner"
    HALLUCINATION_CHECKER = "hallucination_checker"
    MESSAGEFILTER = "Message_filter"
    MESSAGESUMMARY = "Message_summary"
    ISSUETRACK = "Issue_Track"

    




