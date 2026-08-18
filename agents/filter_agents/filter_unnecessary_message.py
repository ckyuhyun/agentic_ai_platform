import json
import os
from typing import List, Any
from langchain.messages import ToolMessage


from langchain_core.prompts import ChatPromptTemplate
from agentic_ai_platform import logger
from agentic_ai_platform.data_class.prompt_spec import PromptSpec
from agentic_ai_platform.data_class.tool_spec import ToolSpec
from agentic_ai_platform.enum.prompt_type import PromptType
from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.tools.tool_hub import get_current_eligible_tools,  next_attempt_number
from agentic_ai_platform.states.filter_message_state import FilterMessageItem, FilterMessageBatchStateLLM
from agentic_ai_platform.states.tool_state import ToolState
from track_issue_system import prompt_hub
from track_issue_system.agents.state_utils import normalize_state, wrap_state


def filter_out_invalid_messages(messages: List[str]) -> List[dict]:
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

async def classify_messages(node_llm,
                       prompt_template: ChatPromptTemplate,
                       message_texts: List[str],
                       batch_size: int,
                       max_concurrency: int) -> List[FilterMessageItem]:
        """
        Classify each message in message_texts as relevant/not relevant, in chunks of
        batch_size sent concurrently via .batch(). Returned items keep their original
        (global) index into message_texts.
        """
        pre_filtered_messages = filter_out_invalid_messages(message_texts)
        #chunks = [pre_filtered_messages[i:i + batch_size] for i in range(0, len(pre_filtered_messages), batch_size)]

        structured_llm = node_llm.llm_instance.with_structured_output(FilterMessageBatchStateLLM)

        results: List[FilterMessageItem] = []
        for chunk_index, chunk in enumerate(pre_filtered_messages):
            # offset = chunk_index * batch_size
            # joined_str = "\n".join(f"{offset + i}: {text}" for i, text in enumerate(chunk))
            prompt = prompt_template.format_messages(input=chunk)

            try:
                response = await structured_llm.abatch([prompt], config={"max_concurrency": max_concurrency})
            except Exception as e:
                # A single chunk failing (e.g. the model's completion got cut off
                # before it could finish the JSON) shouldn't discard results
                # already collected from other chunks.
                logger.error(f'classify_messages chunk {chunk_index} error => {e}')
                continue

            for batch_result in response:
                for llm_item in batch_result.items:
                    if not (0 <= llm_item.index <= len(pre_filtered_messages)):
                        continue
                    # index from llm seems starting with 1
                    message_index = llm_item.index-1 
                    results.append(FilterMessageItem(
                        index=llm_item.index,
                        scoring=llm_item.scoring,
                        reasoning=llm_item.reasoning,
                        cleaned_message=pre_filtered_messages[message_index],
                    ))

        return results


def create_message_filter_agent(node_llm : LLM,
                                tool_llm : LLM, 
                                prompt_template: ChatPromptTemplate,
                                tools: List[ToolSpec] | None = [], 
                                batch_size: int = 5,
                                max_concurrency: int = 4):

  

    # def seed_dataset_to_db(thread_id: str, 
    #                        all_items: List[FilterMessageItem]):
    #     """
    #     The filered messages are logged to the database for future fine-tuning. 
    #     This function attempts to log the filtered messages to the database, 
    #     and any exceptions during this process are caught and logged without interrupting the main flow.
    #     """
    #     if os.getenv("POSTGRES_Dataset_Update", "true") == "true":
    #         try:
    #             from track_issue_system.finetune.db import store_filtered_message_db
    #             store_filtered_message_db(model_name=node_llm.model_name,
    #                                       prompt_version=prompt_version,
    #                                       thread_id=thread_id,
    #                                       items=all_items)
    #         except Exception as e:
    #             logger.error(f'message_filter_agent logging error => {e}')

    #     else:
    #         logger.info("POSTGRES_Dataset_Update is set to false, skipping logging to database.")


    

    async def message_filter_agent(state):
        """
        This function filters out unnecessary messages from the tool states in the given state.
        It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
        The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
        """

        logger.info("message filter agent run")

        state_model, original_was_model = normalize_state(state)


        messages = state.messages[-1]
        if isinstance(messages, ToolMessage):
            messages = messages.content

        
        message_texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in messages]

        if not message_texts:
            logger.info("[message_filter_agent] => No meessages passed")
            return state.model_copy(update={"messages": json.dumps([]),
                                            "messages_filtered":True})

        all_items = await classify_messages(node_llm,
                                        prompt_template,
                                        message_texts,
                                        batch_size=batch_size,
                                        max_concurrency=max_concurrency)

     
        # feed the data into database for future fine-tuning
        if tools:
            tool_llm.bind_tools([spec.tool for spec in tools],
                                tool_required=True)

            eligible_tool_spec = next(t for t in get_current_eligible_tools(tool_specs= tools,
                                                            tool_states=state.tool_states))
            active_tool_spec = next(t for t in tools if t.name == eligible_tool_spec.name)

            
            # Snapshot of state as it will look once this node's results land --
            # InjectedState-annotated tool params (thread_id, filtered_message)
            # are resolved off of this, since all_items isn't part of `state` yet.
            seed_state = state_model.model_copy(update={"filtered_message": all_items})

            human_vars = active_tool_spec.build_human_vars(state_model) if active_tool_spec.build_human_vars else {}
            human_vars = {**human_vars, "messages": all_items, "thread_id": state_model.thread_id}
            prompt_messages = active_tool_spec.prompt_template.format_messages(**human_vars)

            response = await tool_llm.invoke_by_single_prompt(system_human_message=prompt_messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            problems = []
            new_messages = []

            if not tool_calls:
                problems.append("Not have any tool to be called")

            # # argument validation
            for call in tool_calls:
            #     _call = call[0] if isinstance(call, List) else call
            #     spec = next(t for t in tools if call['name'] == t.name)

            #     if spec is None:
            #         problems.append(f"Unknown tool '{_call['name']}'")
            #     elif spec.tool.tool_call_schema is not None:
            #         try:
            #             spec.tool.tool_call_schema.model_validate(call['args'])
            #         except Exception as e:
            #             problems.append(f"Invalid args for '{_call['name']}': {e}")
                #tool_args = inject_state_args(active_tool_spec.tool, call['args'], seed_state)
                tool_args = human_vars.copy()
                
                try:
                    attempt = next_attempt_number(active_tool_spec.name,
                                                            state.tool_states)

                    tool_result = await active_tool_spec.tool.ainvoke(tool_args)
                    tool_state = ToolState(
                                        query=call.get("query",""),
                                        tool_name = active_tool_spec.name,
                                        tool_args = tool_args,
                                        tool_result = tool_result,
                                        status="success",
                                        attempt=attempt
                                    )
                except Exception as e:
                    logger.warning("planner_agent: tool %r failed (attempt %d): %s", active_tool_spec.name, attempt, e)
                    tool_state = ToolState(
                        query=call.get("query",""),
                        tool_name = active_tool_spec.name,
                        tool_args = tool_args,
                        tool_result=None,
                        status="failed",
                        attempt=attempt,
                        error=str(e))

                if tool_state.status == "success":
                    if isinstance(tool_state.tool_result, list):
                        tool_content = [m.get('text', "") if isinstance(m, dict) else m for m in tool_state.tool_result]
                    else:
                        tool_content = str(tool_state.tool_result)

                    new_messages.append(ToolMessage(
                        content=tool_content,
                        tool_call_id=call["id"],
                        name=call["name"]))
            
            updated_node_trace = state.node_traces.copy()
        
            updated_state = state.model_copy(update={
                                    "tool_states": state.tool_states + [tool_state],
                                    "node_traces": updated_node_trace,
                                    "filtered_message" : all_items,
                                    "messages": new_messages,
                                    "messages_filtered" : True
                                })
        
        else:
            updated_node_trace = state.node_traces.copy()
            updated_state = state.model_copy(update={
                        "filtered_message" : all_items,
                        "node_traces": updated_node_trace,
                        "messages_filtered" : True
                    })
            

        return wrap_state(updated_state, original_was_model)
        

    return message_filter_agent
