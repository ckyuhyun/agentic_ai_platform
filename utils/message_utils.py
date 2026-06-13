from typing import List, Tuple
from langchain_core.messages import AnyMessage


def extract_new_messages(state) -> Tuple[List[AnyMessage], int]:
    """Return the messages appended since `state.last_reviewed_message_index`,
    along with the cursor value the caller should store after processing them.
    """
    messages = state.messages
    new_messages = messages[state.last_reviewed_message_index:]
    return new_messages, len(messages)
