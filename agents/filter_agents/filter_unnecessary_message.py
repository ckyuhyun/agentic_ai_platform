import json
import os
from typing import List
from dotenv import load_dotenv
from langchain.messages import ToolMessage


from langchain_core.prompts import ChatPromptTemplate
from agentic_ai_platform import logger
from agentic_ai_platform.states.filter_message_state import FilterMessageBatchState, FilterMessageItem
from track_issue_system.finetune.db import filtered_message_log_predictions


load_dotenv()

def filter_out_unnecessary_messages(messages: List[str]) -> List[dict]:
    """
    Filters out unnecessary messages from the given list of messages.
    Unnecessary messages are those that are empty or contain only whitespace.
    
    Args:
        messages (List[str]): A list of message strings.
    """
    # 1. filter out messages that are empty or contain only whitespace
    filtered_messages = [msg for msg in messages if msg.strip()]

    # 2. filter out messages that are repetitive or duplicates based on their text content
    seen_texts = set(msg.strip() for msg in filtered_messages)

    # 3. filter out messages that might be too short to be relevant (e.g., less than 10 characters)
    filtered_messages = [msg for msg in list(seen_texts) if len(msg.strip()) > 10]

    
    return filtered_messages

def classify_messages(node_llm,
                       prompt_template: ChatPromptTemplate,
                       message_texts: List[str],
                       batch_size: int,
                       max_concurrency: int) -> List[FilterMessageItem]:
        """
        Classify each message in message_texts as relevant/not relevant, in chunks of
        batch_size sent concurrently via .batch(). Returned items keep their original
        (global) index into message_texts.
        """
        pre_filtered_messages = filter_out_unnecessary_messages(message_texts)
        chunks = [pre_filtered_messages[i:i + batch_size] for i in range(0, len(pre_filtered_messages), batch_size)]

        try:
            
            results = []
            for chunk_index, chunk in enumerate(chunks):
                offset = chunk_index * batch_size
                joined_str = "\n".join(f"{offset + i}: {text}" for i, text in enumerate(chunk))
                prompt = prompt_template.format_messages(input=joined_str)

                structured_llm = node_llm.llm_instance.with_structured_output(FilterMessageBatchState)
                results.extend(structured_llm.batch([prompt], config={"max_concurrency": max_concurrency}))
            
            
            return [item for result in results for item in result.items]
        except Exception as e:
            logger.error(f'classify_messages error => {e}')
            return []


def create_message_filter_agent(node_llm,
                                system_prompt: str,
                                batch_size: int = 5,
                                max_concurrency: int = 4,
                                prompt_version: str = "0"):

    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), 
                                                        ("human", "{input}")])


    def seed_dataset_to_db(thread_id: str, 
                           all_items: List[FilterMessageItem]):
        """
        The filered messages are logged to the database for future fine-tuning. 
        This function attempts to log the filtered messages to the database, 
        and any exceptions during this process are caught and logged without interrupting the main flow.
        """
        if os.getenv("POSTGRES_Dataset_Update", "true") == "true":
            try:
                from track_issue_system.finetune.db import filtered_message_log_predictions
                filtered_message_log_predictions(model_name=node_llm.model_name,
                                                prompt_version=prompt_version,
                                                thread_id=thread_id,
                                                items=all_items)
            except Exception as e:
                logger.error(f'message_filter_agent logging error => {e}')

        else:
            logger.info("POSTGRES_Dataset_Update is set to false, skipping logging to database.")


    

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

            all_items = classify_messages(node_llm, 
                                          prompt_template, 
                                          message_texts,
                                          batch_size=batch_size, 
                                          max_concurrency=max_concurrency)

            # feed the data into database for future fine-tuning
            seed_dataset_to_db(state.thread_id,
                               all_items)
        except Exception as e:
            logger.error(f'message_filter_agent error => {e}')
            return state.model_copy(update={"filtered_messages": [],
                                             "messages_filtered": False})

      

        # relevant_indices = sorted(
        #     item.index for item in all_items
        #     if item.not_cleaned_message and 0 <= item.index < len(message_texts)
        # )
        # final_cleaned_messages = [message_texts[i] for i in relevant_indices]

        filtered = FilterMessageBatchState(items=all_items)
        
        return state.model_copy(update={"filtered_message": filtered, 
                                        "messages_filtered": True})
        

    return message_filter_agent
