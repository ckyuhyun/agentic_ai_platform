

from langchain.messages import ToolMessage
import json

from langchain_core.prompts import ChatPromptTemplate

from track_issue_system.State.filter_message_state import FilterMessageBatchState


def create_message_filter_agent(node_llm,
                                system_prompt: str):

    def message_filter_agent(state):
        """
        This function filters out unnecessary messages from the tool states in the given state.
        It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
        The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
        """

        messages = state.messages[-1]
        if isinstance(messages, ToolMessage):
            messages = messages.content

        message_texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in messages]

        if not message_texts:
            return state.model_copy(update={"messages": json.dumps([])})

        joined_str = "\n".join(f"{i}: {text}" for i, text in enumerate(message_texts))

        prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        prompt = prompt_template.format_messages(input=joined_str)
        prompt[0].additional_kwargs["cache_control"] = {"type": "ephemeral"}

        result = node_llm.llm_instance.with_structured_output(FilterMessageBatchState).invoke(prompt)

        # relevant_indices = sorted(
        #     item.index for item in result.items
        #     if item.not_cleaned_message and 0 <= item.index < len(message_texts)
        # )
        # final_cleaned_messages = [message_texts[i] for i in relevant_indices]

        #return state.model_copy(update={"messages": json.dumps(final_cleaned_messages)})
        return state

    return message_filter_agent
