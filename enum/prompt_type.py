from enum import Enum

class PromptType(str, Enum):
    REWRITE = "rewrite"
    REWRITE_EVAL = "rewrite_eval"
    EXECUTE_TOOLS = "execute_tools"
    CRITIC = "critic"
    DRAFTER = "drafter"
    PLANNER = "planner"
    DADTAINGESTION = "DataIngestion"
    HALLUCINATION_CHECKER = "hallucination_checker"
    MESSAGEFILTER = "Message_filter"
    VectorSearch = "Vector_Search"
    MESSAGESUMMARY = "Message_summary"
    ISSUETRACK = "Issue_Track"
    INJECTION_GUARD = "injection_guard"
    HUMAN_REVIEW = "human_review"
    QUERY_Rewrite = "query_clarifier"

    




