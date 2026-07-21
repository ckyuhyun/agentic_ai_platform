

from langchain.messages import ToolMessage
import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from track_issue_system.State.filter_message_state import FilterMessageBatchState, FilterMessageItem

import logging

logger = logging.getLogger(__name__)


def classify_messages(node_llm,
                       prompt_template: ChatPromptTemplate,
                       message_texts: List[str],
                       batch_size: int = 20,
                       max_concurrency: int = 4) -> List[FilterMessageItem]:
    """
    Classify each message in message_texts as relevant/not relevant, in chunks of
    batch_size sent concurrently via .batch(). Returned items keep their original
    (global) index into message_texts.
    """

    chunks = [message_texts[i:i + batch_size] for i in range(0, len(message_texts), batch_size)]

    prompts = []
    for chunk_index, chunk in enumerate(chunks):
        offset = chunk_index * batch_size
        joined_str = "\n".join(f"{offset + i}: {text}" for i, text in enumerate(chunk))
        prompts.append(prompt_template.format_messages(input=joined_str))

    structured_llm = node_llm.llm_instance.with_structured_output(FilterMessageBatchState)
    results = structured_llm.batch(prompts, 
                                   config={"max_concurrency": max_concurrency})
    return [item for result in results for item in result.items]


def create_message_filter_agent(node_llm,
                                system_prompt: str,
                                batch_size: int = 20,
                                max_concurrency: int = 4):

    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    def message_filter_agent(state):
        """
        This function filters out unnecessary messages from the tool states in the given state.
        It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
        The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
        """

        logger.info("message filter agent run")

        messages = state.messages[-1]
        if isinstance(messages, ToolMessage):
            messages = messages.content

        try:
            message_texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in messages]

            if not message_texts:
                logger.info("[message_filter_agent] => No meessages passed")
                return state.model_copy(update={"messages": json.dumps([]),
                                                "messages_filtered":True})

            all_items = classify_messages(node_llm, prompt_template, message_texts,
                                        batch_size=batch_size, max_concurrency=max_concurrency)
        except Exception as e:
            logger.error(f'message_filter_agent error => {e}')
            return state.model_copy(update={"filtered_messages": [], "messages_filtered": False})

        # relevant_indices = sorted(
        #     item.index for item in all_items
        #     if item.not_cleaned_message and 0 <= item.index < len(message_texts)
        # )
        # final_cleaned_messages = [message_texts[i] for i in relevant_indices]

        filtered = FilterMessageBatchState(items=all_items)
        
        return state.model_copy(update={"filtered_message": filtered, 
                                        "messages_filtered": True})
        

    return message_filter_agent
