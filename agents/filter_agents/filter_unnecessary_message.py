import json
import os
from typing import List, Any
from langchain.messages import ToolMessage, HumanMessage


from langchain_core.prompts import ChatPromptTemplate
from agentic_ai_platform import logger
from agentic_ai_platform.data_class.prompt_spec import PromptSpec
from agentic_ai_platform.data_class.tool_spec import ToolSpec
from agentic_ai_platform.enum.prompt_type import PromptType
from agentic_ai_platform.llm.llm import LLM
from agentic_ai_platform.tools.tool_hub import get_current_eligible_tools,  next_attempt_number
from agentic_ai_platform.states.filter_message_state import FilterMessageItem, FilterMessageItemLLM
from agentic_ai_platform.states.tool_state import ToolState
from agentic_ai_platform.graph.node_trace import NodeTrace


from track_issue_system.agents.state_utils import normalize_state, wrap_state


_MAX_SELF_CORRECTION_ATTEMPTS = 2


async def _invoke_tool_with_self_correction(tool_llm: LLM,
                                             prompt_messages: List[Any],
                                             active_tool_spec: ToolSpec) -> List[dict]:
    """
    Invoke tool_llm forcing a call to active_tool_spec, re-prompting up to
    _MAX_SELF_CORRECTION_ATTEMPTS times if the model fails to produce a
    valid tool call. Small local models forced (via tool_choice="required")
    into emitting a call with a nontrivial nested-argument schema (e.g.
    seed_dataset_to_db's List[FilterMessageItem]) don't always comply on the
    first try -- see planner_agent.py's _invoke_tool_with_self_correction
    for the same pattern.
    """
    messages = list(prompt_messages)
    tool_calls: List[dict] = []

    for correction_attempt in range(1, _MAX_SELF_CORRECTION_ATTEMPTS + 1):
        response = await tool_llm.invoke_by_single_prompt(system_human_message=messages)

        # invoke_by_single_prompt returns a single AIMessage-like response
        # when the prompt fits within TOKEN_LIMIT, but a List of responses
        # (one per chunk) when it had to fall back to _batch_invoke -- flatten
        # both shapes into one list of tool calls instead of silently
        # dropping them.
        if hasattr(response, "tool_calls"):
            tool_calls = getattr(response, "tool_calls", None) or []
        else:
            tool_calls = [call for r in (response or []) for call in (getattr(r, "tool_calls", None) or [])]

        problems = []
        if not tool_calls:
            problems.append("Not have any tool to be called")

        for call in tool_calls:
            if call["name"] != active_tool_spec.name:
                problems.append(f"Unknown tool '{call['name']}'; expected '{active_tool_spec.name}'.")
            elif active_tool_spec.tool.tool_call_schema is not None:
                try:
                    active_tool_spec.tool.tool_call_schema.model_validate(call["args"])
                except Exception as e:
                    problems.append(f"Invalid args for '{call['name']}': {e}")

        if not problems:
            return tool_calls

        logger.warning("message_filter_agent: self-correction attempt %d/%d failed: %s",
                        correction_attempt, _MAX_SELF_CORRECTION_ATTEMPTS, problems)
        messages = messages + [HumanMessage(
            content="Your previous response was invalid: " + " ".join(problems) +
                    f" Call the '{active_tool_spec.name}' tool with valid arguments."
        )]

    logger.warning("message_filter_agent: giving up on tool '%s' after %d self-correction attempts",
                    active_tool_spec.name, _MAX_SELF_CORRECTION_ATTEMPTS)
    return tool_calls


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
        
        pre_filtered_messages = filter_out_invalid_messages(set(message_texts))
        #chunks = [pre_filtered_messages[i:i + batch_size] for i in range(0, len(pre_filtered_messages), batch_size)]

        structured_llm = node_llm.llm_instance.with_structured_output(FilterMessageItemLLM)

        results: List[FilterMessageItem] = []
        prompts = [
            prompt_template.format_messages(input=f'{chunk_index} : {chunk}')
            for chunk_index, chunk in enumerate(pre_filtered_messages)
        ]

        try:
            response = await structured_llm.abatch(prompts, config={"max_concurrency": max_concurrency})
        except Exception as e:
            # A single chunk failing (e.g. the model's completion got cut off
            # before it could finish the JSON) shouldn't discard results
            # already collected from other chunks.
            logger.error(f'classify_messages error => {e}')
            response = []


        for index, batch_result in enumerate(response):
                # index from llm seems starting with 
                message_index = index
                results.append(FilterMessageItem(
                    index=message_index,
                    scoring=batch_result. scoring,
                    reasoning=batch_result.reasoning,
                    cleaned_message=pre_filtered_messages[message_index],
                ))

        return results


def create_message_filter_agent(node_llm : LLM,
                                tool_llm : LLM, 
                                prompt_template: ChatPromptTemplate,
                                tools: List[ToolSpec] | None = [], 
                                batch_size: int = 5,
                                max_concurrency: int = 4):

    async def message_filter_agent(state):
        """
        This function filters out unnecessary messages from the tool states in the given state.
        It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
        The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
        """

        logger.info("message filter agent run")

        state_model, original_was_model = normalize_state(state)

        trace = NodeTrace.start(node="message_filter_agent", 
                                        iteration=state_model.iteration, 
                                        model=node_llm.model_name)

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

            tool_calls = await _invoke_tool_with_self_correction(tool_llm=tool_llm,
                                                                   prompt_messages=prompt_messages,
                                                                   active_tool_spec=active_tool_spec)

            new_messages = []

            if not tool_calls:
                logger.warning("message_filter_agent: no valid tool call for '%s' after self-correction; "
                                "skipping tool invocation for thread_id=%s",
                                active_tool_spec.name, state_model.thread_id)

            tool_states : List[ToolState] = []
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

                tool_states.append(tool_state)

                if tool_state.status == "success":
                    if isinstance(tool_state.tool_result, list):
                        tool_content = [m.get('text', "") if isinstance(m, dict) else m for m in tool_state.tool_result]
                    else:
                        tool_content = str(tool_state.tool_result)

                    new_messages.append(ToolMessage(
                        content=tool_content,
                        tool_call_id=call["id"],
                        name=call["name"]))

            # Trace node update
            updated_node_trace = state.node_traces.copy()
            updated_node_trace.append(trace.finish(tool_calls_made = [c["name"] for c in tool_calls]))
            
            
            # Update state
            updated_state = state.model_copy(update={
                                    "tool_states": state.tool_states + tool_states,
                                    "node_traces": updated_node_trace,
                                    "filtered_message" : all_items,
                                    "messages": new_messages,
                                    "messages_filtered" : True
                                })
    
        else:
            # Trace node update
            updated_node_trace = state.node_traces.copy()
            
            # Update state
            updated_state = state.model_copy(update={
                        "filtered_message" : all_items,
                        "node_traces": updated_node_trace,
                        "messages_filtered" : True
                    })
            

        return wrap_state(updated_state, original_was_model)
        

    return message_filter_agent
