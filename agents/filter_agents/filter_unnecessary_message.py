

from langchain.messages import ToolMessage
import ast
import json


def filter_unnecessary_messages(state):
    """
    This function filters out unnecessary messages from the tool states in the given state.
    It checks each tool state and removes those that are deemed unnecessary based on certain criteria.
    The criteria for filtering can be defined as needed, such as removing tool states with empty results or those that do not contribute to the final output.
    """
    
    
    black_list = {}
    
    messages = state.messages
    final_cleaned_messages :str = ""
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.content and "get_slack_message" in msg.name:
                try:
                    list_messages = ast.literal_eval(msg.content)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid content in message: {msg.content}, error: {e}")

                cleaned_messages : list[str]= []
                for message in list_messages:
                    if not message.get("text", "") in black_list:
                        cleaned_messages.append(message)

                final_cleaned_messages = json.dumps(cleaned_messages)
                        
                
    return state.model_copy(update={"messages": final_cleaned_messages})